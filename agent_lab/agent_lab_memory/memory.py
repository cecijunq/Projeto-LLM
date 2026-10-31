"""
Memória de longo prazo para o Agent Laboratory.

O agente original só tem memória de curto prazo (o histórico do prompt, com
decaimento) e descarta tudo ao final de cada projeto. Aqui adicionamos uma
memória persistente entre projetos, inspirada em Generative Agents (Park et al.,
2023), Reflexion (Shinn et al., 2024) e ExpeL (Zhao et al., 2024).

Tipos de memória (campo `kind`):
  - episodic   : resumo do que aconteceu em cada fase de um projeto (plano, resultados, notas)
  - insight    : lições generalizáveis extraídas por reflexão no fim do projeto
  - error      : erros de código encontrados pelo mle-solver e como foram corrigidos
  - literature : artigos do arXiv já lidos e julgados relevantes (cache semântico)
  - feedback   : notas do humano no modo co-pilot (preferências do pesquisador)

Recuperação (retrieve): pontuação = w_sim * similaridade + w_imp * importância + w_rec * recência
  - similaridade : cosseno entre embeddings (sentence-transformers se instalado,
                   senão um bag-of-words com TF-IDF em Python puro)
  - importância  : 0..1, dada na escrita (ex.: lições de reviews ruins pesam mais)
  - recência     : decay ** (projetos decorridos desde o último acesso)
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict

from .llm import extract_command

KINDS = ("episodic", "insight", "error", "literature", "feedback")

# Quais tipos de memória cada fase consulta.
PHASE_KINDS = {
    "literature_review": ["literature", "insight", "feedback"],
    "plan_formulation": ["insight", "episodic", "feedback"],
    "data_preparation": ["insight", "error", "feedback"],
    "running_experiments": ["insight", "error", "episodic", "feedback"],
    "results_interpretation": ["insight", "episodic"],
    "report_writing": ["insight", "feedback"],
    "report_refinement": ["insight"],
}


@dataclass
class MemoryItem:
    content: str
    kind: str
    topic: str
    phase: str | None = None
    importance: float = 0.5
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_run: int = 0
    last_access_run: int = 0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- embeddings
_TOKEN = re.compile(r"[a-zA-ZÀ-ÿ0-9_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text) if len(t) > 2]


class Embedder:
    """Usa sentence-transformers se disponível; caso contrário, TF-IDF esparso."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_dense: bool = True):
        self.dense = None
        if use_dense:
            try:
                from sentence_transformers import SentenceTransformer
                self.dense = SentenceTransformer(model_name)
            except Exception:
                self.dense = None

    def similarities(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        if self.dense is not None:
            vecs = self.dense.encode([query] + docs, normalize_embeddings=True)
            q = vecs[0]
            return [float(max(0.0, q @ v)) for v in vecs[1:]]
        # TF-IDF em Python puro
        tok_docs = [Counter(_tokens(d)) for d in docs]
        q_tok = Counter(_tokens(query))
        n = len(docs) + 1
        df = Counter()
        for c in tok_docs + [q_tok]:
            df.update(c.keys())
        idf = {t: math.log(n / df[t]) + 1.0 for t in df}

        def vec(c):
            v = {t: f * idf[t] for t, f in c.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return {t: x / norm for t, x in v.items()}

        qv = vec(q_tok)
        return [sum(qv.get(t, 0.0) * x for t, x in vec(c).items()) for c in tok_docs]


# ----------------------------------------------------------------------------- store
class LongTermMemory:
    def __init__(self, path: str = "memory_store.json", embedder: Embedder | None = None,
                 w_sim: float = 0.6, w_imp: float = 0.25, w_rec: float = 0.15,
                 recency_decay: float = 0.9, dedup_threshold: float = 0.9,
                 max_items_per_kind: int = 200, read_only: bool = False):
        self.path = path
        self.read_only = read_only  # True na fase de TESTE: a memória é consultada mas nunca alterada
        self.embedder = embedder or Embedder()
        self.w_sim, self.w_imp, self.w_rec = w_sim, w_imp, w_rec
        self.recency_decay = recency_decay
        self.dedup_threshold = dedup_threshold
        self.max_items_per_kind = max_items_per_kind
        self.items: list[MemoryItem] = []
        self.run_counter = 0  # quantos projetos já passaram por esta memória
        self.retrieval_log: list[dict] = []  # para análise: o que foi recuperado em cada fase
        self.load()

    # --- persistência
    def load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.run_counter = data.get("run_counter", 0)
            self.items = [MemoryItem(**d) for d in data.get("items", [])]

    def save(self):
        if self.read_only:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"run_counter": self.run_counter,
                       "items": [asdict(i) for i in self.items]}, f, ensure_ascii=False, indent=2)

    def start_run(self):
        if not self.read_only:
            self.run_counter += 1
        self.retrieval_log = []

    # --- escrita
    def add(self, content: str, kind: str, topic: str, phase: str | None = None,
            importance: float = 0.5, **metadata) -> MemoryItem | None:
        assert kind in KINDS, kind
        if self.read_only:
            return None
        content = content.strip()
        same_kind = [i for i in self.items if i.kind == kind]
        if same_kind:  # deduplicação: memória quase igual → reforça em vez de duplicar
            sims = self.embedder.similarities(content, [i.content for i in same_kind])
            best = max(range(len(sims)), key=sims.__getitem__)
            if sims[best] >= self.dedup_threshold:
                it = same_kind[best]
                it.importance = min(1.0, max(it.importance, importance) + 0.05)
                it.last_access_run = self.run_counter
                it.metadata.setdefault("reinforced", 0)
                it.metadata["reinforced"] += 1
                return it
        it = MemoryItem(content=content, kind=kind, topic=topic, phase=phase,
                        importance=float(max(0.0, min(1.0, importance))),
                        created_run=self.run_counter, last_access_run=self.run_counter,
                        metadata=metadata)
        self.items.append(it)
        self._evict(kind)
        return it

    def _evict(self, kind: str):
        same = [i for i in self.items if i.kind == kind]
        if len(same) <= self.max_items_per_kind:
            return
        same.sort(key=lambda i: i.importance * self._recency(i))
        drop = {i.id for i in same[: len(same) - self.max_items_per_kind]}
        self.items = [i for i in self.items if i.id not in drop]

    # --- leitura
    def _recency(self, it: MemoryItem) -> float:
        return self.recency_decay ** max(0, self.run_counter - it.last_access_run)

    def retrieve(self, query: str, kinds: list[str] | None = None, phase: str | None = None,
                 k: int = 6, min_sim: float = 0.05) -> list[MemoryItem]:
        pool = [i for i in self.items if (kinds is None or i.kind in kinds)]
        if not pool:
            return []
        sims = self.embedder.similarities(query, [f"{i.topic}. {i.content}" for i in pool])
        scored = []
        for it, sim in zip(pool, sims):
            # lições são gerais por definição: as da PRÓPRIA fase sempre concorrem, mesmo com pouca
            # semelhança textual com o tema (senão o filtro descartaria justamente o que é útil)
            same_phase = phase is not None and it.phase == phase
            if sim < min_sim and it.kind != "feedback" and not same_phase:
                continue
            phase_bonus = 0.1 if (phase and it.phase == phase) else 0.0
            score = self.w_sim * sim + self.w_imp * it.importance + self.w_rec * self._recency(it) + phase_bonus
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [it for _, it in scored[:k]]
        for it in chosen:
            it.access_count += 1
            it.last_access_run = self.run_counter
        self.retrieval_log.append({"phase": phase, "query": query[:200], "ids": [i.id for i in chosen]})
        return chosen

    def context_for_phase(self, phase: str, topic: str, extra_query: str = "", k: int = 6) -> str:
        items = self.retrieve(f"{topic}\n{extra_query}", kinds=PHASE_KINDS.get(phase), phase=phase, k=k)
        return format_memories(items)

    def stats(self) -> dict:
        c = Counter(i.kind for i in self.items)
        return {"runs": self.run_counter, "total": len(self.items), **{k: c.get(k, 0) for k in KINDS}}


def format_memories(items: list[MemoryItem]) -> str:
    if not items:
        return ""
    labels = {"episodic": "projeto anterior", "insight": "lição", "error": "erro conhecido",
              "literature": "artigo já lido", "feedback": "preferência do pesquisador"}
    lines = ["=== MEMÓRIA DE LONGO PRAZO (experiência de projetos anteriores) ===",
             "Use estas informações se forem úteis; ignore-as se não se aplicarem ao tópico atual."]
    for it in items:
        lines.append(f"- [{labels[it.kind]} | tópico: {it.topic[:60]}] {it.content}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- lições
PHASES_LESSON = ["literature_review", "plan_formulation", "data_preparation", "running_experiments",
                 "results_interpretation", "report_writing"]

LESSON_SYSTEM = (
    "You extract reusable lessons from feedback on a finished automated ML research project, so that FUTURE "
    "projects on OTHER topics avoid the same problems. Each lesson states what went wrong and how to avoid it. "
    "Base the lessons ONLY on the feedback and the project facts given; never invent problems that are not "
    "mentioned. Lessons must be general (not specific to this topic), concrete and actionable, written in English "
    "even if the feedback is in another language. Return ONLY a JSON list inside ```json ... ``` where each element is "
    '{"problem": str, "how_to_avoid": str, "phase": one of ' + json.dumps(PHASES_LESSON) + ', "importance": float 0..1}. '
    "Higher importance for problems that hurt the result most. At most 4 lessons. "
    "(Task id: extract reusable lessons.)"
)

SOURCE_LABEL = {"llm": "automated LLM reviewer (NeurIPS-style review)",
                "human": "human researchers who read the report"}


def project_facts(result: dict, max_chars: int = 2500) -> str:
    """Fatos objetivos do projeto (iguais para as condições B e C): resultado das fases, falhas, erros."""
    tr = result.get("trace", {})
    fails = result.get("failures", [])
    fail_txt = "; ".join(f"{f['fase']}:{f['categoria']}" for f in fails) or "none detected"
    errs = "\n".join(e.get("text", "")[-300:] for e in tr.get("errors", [])[:4])
    txt = (f"Phase success: {json.dumps(result.get('metrics', {}).get('phase_success', {}))}\n"
           f"Automatically detected failures: {fail_txt}\n"
           f"Plan (excerpt): {tr.get('plan', '')[:600]}\n"
           f"Final experiment output (excerpt): {tr.get('exp_results', '')[-600:]}\n"
           f"Code errors (sample):\n{errs}")
    return txt[:max_chars]


def extract_lessons(llm, topic: str, facts: str, feedback: str, source: str) -> list[dict]:
    """Transforma feedback (do revisor LLM ou de humanos) em lições 'problema -> como evitar'.
    O MESMO extrator é usado para B e C; só muda o texto do feedback (QP2)."""
    prompt = (f"Research topic: {topic}\n\nPROJECT FACTS:\n{facts}\n\n"
              f"FEEDBACK (source: {SOURCE_LABEL.get(source, source)}):\n{feedback}\n\nNow extract the lessons.")
    raw = llm.query(LESSON_SYSTEM, prompt, temperature=0.2)
    body = extract_command(raw, "json") or raw
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", body, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            data = []
    lessons = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        prob = d.get("problem") or d.get("lesson")
        if not prob:
            continue
        lessons.append({"problem": str(prob).strip(), "how_to_avoid": str(d.get("how_to_avoid", "")).strip(),
                        "phase": d.get("phase") if d.get("phase") in PHASES_LESSON else None,
                        "importance": float(d.get("importance", 0.5) or 0.5)})
    return lessons[:4]


def store_lessons(memory: LongTermMemory, lessons: list[dict], topic: str, source: str, project: str) -> int:
    n = 0
    for les in lessons:
        content = f"Problem: {les['problem']} | How to avoid: {les['how_to_avoid']}".strip(" |")
        if memory.add(content, "insight", topic, phase=les["phase"], importance=les["importance"],
                      source=source, project=project) is not None:
            n += 1
    return n


def format_llm_review(review: dict) -> str:
    """Feedback do revisor automático em texto (entrada da condição B)."""
    if not review:
        return "The automated reviewer produced no review (the project did not reach the report stage)."
    keys = ["Overall", "Soundness", "Presentation", "Contribution", "Quality", "Clarity", "Significance"]
    scores = ", ".join(f"{k}={review[k]}" for k in keys if k in review)
    parts = [f"Scores: {scores}. Decision: {review.get('Decision', '?')}"]
    for k in ("Weaknesses", "Questions", "Limitations"):
        v = review.get(k)
        if v:
            parts.append(f"{k}: " + ("; ".join(map(str, v)) if isinstance(v, list) else str(v)))
    return "\n".join(parts)


def consolidate_run(memory: LongTermMemory, llm, result: dict, project: str = "") -> int:
    """Modo 'online' (opcional): ao fim de cada projeto, lições do revisor LLM entram direto na memória."""
    if memory.read_only:
        return 0
    lessons = extract_lessons(llm, result["topic"], project_facts(result), format_llm_review(result.get("review")), "llm")
    n = store_lessons(memory, lessons, result["topic"], "llm", project)
    memory.save()
    return n