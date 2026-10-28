"""
Orquestrador do fluxo (Figura 2 do artigo) com memória de longo prazo opcional.

Fases: literature_review -> plan_formulation -> data_preparation -> running_experiments
       -> results_interpretation -> report_writing -> report_refinement

Onde a memória entra (# [MEMÓRIA]):
  LEITURA : no início de cada fase, recupera as lições relevantes ao tópico/fase e injeta no prompt.
  ESCRITA : só acontece se a memória NÃO for somente-leitura. No protocolo do projeto (projeto.py)
            a memória é construída offline a partir do feedback (LLM ou humano) e fica congelada
            durante os testes, então nada é escrito aqui.
Com memory=None o comportamento é o do Agent Laboratory original (versão A).
"""
from __future__ import annotations

import json
import os
import re
import time

from .agents import Agent
from .llm import extract_command
from .failures import detect_failures
from .memory import LongTermMemory, consolidate_run
from .solvers import MLESolver, PaperSolver
from .tools import ArxivSearch, execute_code, search_hf_datasets

PHASES = ["literature_review", "plan_formulation", "data_preparation", "running_experiments",
          "results_interpretation", "report_writing", "report_refinement"]


class AgentLaboratory:
    def __init__(self, topic: str, llm, memory: LongTermMemory | None = None, copilot: bool = False,
                 notes: dict | None = None, num_papers: int = 5, max_steps: int = 20,
                 mle_steps: int = 3, paper_steps: int = 2, max_revisits: int = 1,
                 exp_timeout: int = 600, out_dir: str = "runs/run", verbose: bool = True):
        self.topic, self.llm, self.memory = topic, llm, memory
        self.copilot, self.notes = copilot, notes or {}
        self.num_papers, self.max_steps = num_papers, max_steps
        self.mle_steps, self.paper_steps, self.max_revisits = mle_steps, paper_steps, max_revisits
        self.exp_timeout, self.out_dir, self.verbose = exp_timeout, out_dir, verbose
        self.phd, self.postdoc = Agent("phd", llm, notes=self.notes), Agent("postdoc", llm, notes=self.notes)
        self.mle = Agent("mle", llm, notes=self.notes)
        self.arxiv = ArxivSearch()
        os.makedirs(out_dir, exist_ok=True)
        self.llm.log_path = os.path.join(out_dir, "llm_calls.jsonl")  # respostas brutas, para diagnóstico
        self.last_response: dict[str, str] = {}
        # estado do projeto
        self.lit_review: list[dict] = []
        self.plan = self.dataset_code = self.exp_code = self.exp_results = ""
        self.interpretation = self.report = self.report_md = ""
        self.review: dict = {}
        self.mle_score = 0.0
        self.human_feedback: list[str] = []
        self.errors: list[dict] = []   # {"phase": ..., "text": ...}
        self.metrics: dict = {"phase_steps": {}, "phase_success": {}, "phase_time": {}, "invalid_commands": {}}

    # ------------------------------------------------------------------ utilidades
    def log(self, *a):
        if self.verbose:
            print("[AgentLab]", *a, flush=True)

    def mem(self, phase: str, extra: str = "") -> str:
        if self.memory is None:
            return ""
        return self.memory.context_for_phase(phase, self.topic, extra_query=extra)  # [MEMÓRIA] leitura

    def invalid(self, phase: str, resp: str, example: str) -> str:
        """Conta o erro de formato e devolve um feedback que mostra o formato correto."""
        self.metrics["invalid_commands"][phase] = self.metrics["invalid_commands"].get(phase, 0) + 1
        return ("Your last output did not contain a valid command, so nothing happened. "
                f"Reply with exactly one fenced command, for example:\n{example}")

    def lit_str(self) -> str:
        return "\n".join(f"[{p['id']}] {p['summary']}" for p in self.lit_review)

    def checkpoint(self, phase: str, output: str) -> str:
        """Modo co-pilot: o humano revisa a saída da fase; nota vira memória de 'feedback'."""
        if not self.copilot:
            return ""
        print(f"\n===== Saída da fase {phase} =====\n{output[:3000]}\n")
        fb = input("Enter = seguir | ou escreva uma nota para refazer a fase: ").strip()
        if fb:
            self.human_feedback.append(f"{phase}: {fb}")
            if self.memory is not None:  # [MEMÓRIA] preferência do pesquisador, alta importância
                self.memory.add(fb, "feedback", self.topic, phase=phase, importance=0.9)
        return fb

    # ------------------------------------------------------------------ fases
    def literature_review(self) -> bool:
        phase, feedback = "literature_review", ""
        known = self.mem(phase)  # [MEMÓRIA] artigos já lidos em projetos parecidos + lições
        for step in range(self.max_steps):
            resp = self.phd.step(self.topic, phase, step, self.max_steps, feedback,
                                 context=f"Papers added so far ({len(self.lit_review)}/{self.num_papers}):\n{self.lit_str()}",
                                 memory_context=known)
            if (q := extract_command(resp, "SUMMARY")) is not None:
                feedback = f"Paper summaries for '{q}':\n{self.arxiv.summaries(q)[:6000]}"
            elif (pid := extract_command(resp, "FULL_TEXT")) is not None:
                feedback = f"Full text:\n{self.arxiv.full_text(pid)[:8000]}"
            elif (add := extract_command(resp, "ADD_PAPER")) is not None:
                pid, _, summ = add.partition("\n")
                self.lit_review.append({"id": pid.strip(), "summary": summ.strip()})
                feedback = f"Paper {pid.strip()} added."
                if self.memory is not None:  # [MEMÓRIA] cache semântico de literatura
                    self.memory.add(f"arXiv {pid.strip()}: {summ.strip()[:500]}", "literature", self.topic,
                                    phase=phase, importance=0.4, arxiv_id=pid.strip())
            else:
                feedback = self.invalid(phase, resp, "```SUMMARY\nlanguage models differential diagnosis MedQA\n```\n"
                                        "or\n```ADD_PAPER\n2401.01234\nOne-paragraph summary of the paper and its results.\n```")
            self.last_response[phase] = resp
            if len(self.lit_review) >= self.num_papers:
                self.metrics["phase_steps"][phase] = step + 1
                return True
        self.metrics["phase_steps"][phase] = self.max_steps
        return len(self.lit_review) > 0

    def _dialogue(self, phase: str, a: Agent, b: Agent, submit_cmd: str, context: str) -> str | None:
        """Diálogo genérico: 'a' propõe, 'b' dirige e submete com submit_cmd."""
        memory_context = self.mem(phase)  # [MEMÓRIA]
        fb_a = fb_b = reminder = ""
        for step in range(self.max_steps):
            ra = a.step(self.topic, phase, step, self.max_steps, fb_a, context, memory_context)
            fb_b = extract_command(ra, "DIALOGUE") or ra[:1500]
            rb = b.step(self.topic, phase, step, self.max_steps, fb_b, context + reminder, memory_context)
            self.last_response[phase] = rb
            if (sub := extract_command(rb, submit_cmd)) is not None and sub:
                self.metrics["phase_steps"][phase] = step + 1
                return sub
            fb_a = extract_command(rb, "DIALOGUE")
            if fb_a is None:
                fb_a = rb[:1500]
                self.invalid(phase, rb, f"```DIALOGUE\nyour message\n```  or  ```{submit_cmd}\nfinal text\n```")
            if step + 1 >= self.max_steps // 2:  # lembrete explícito para quem precisa submeter
                reminder = (f"\nREMINDER: you are running out of steps. Submit now with "
                            f"```{submit_cmd}\n<final text>\n```\n")
        self.metrics["phase_steps"][phase] = self.max_steps
        return None

    def suggest_datasets(self) -> str:
        """Busca datasets REAIS no HuggingFace antes da preparação de dados.
        Modelos pequenos inventam nomes (ex.: load_dataset('medqa')); com a lista verificada, escolhem um que existe."""
        sys_p = ("Return 3 short HuggingFace dataset search queries (1 to 3 words each) for the experiment below, "
                 "one per line, no numbering, no explanation.")
        raw = self.llm.query(sys_p, f"Topic: {self.topic}\nPlan:\n{self.plan[:1500]}", temperature=0.3)
        queries = []
        for line in raw.splitlines():
            q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("`\"'").strip()
            if q and not q.isupper() and len(q) <= 40 and q.lower() not in ("text", "plaintext"):
                queries.append(q)
        found: list[str] = []
        for q in queries[:3]:
            for line in search_hf_datasets(q, limit=5).splitlines():
                if line.startswith("- ") and line not in found:
                    found.append(line)
        self.metrics["hf_suggestions"] = {"queries": queries[:3], "found": len(found)}
        lst = "\n".join(found[:12]) if found else "(no results; use a torchvision or sklearn built-in dataset)"
        return ("VERIFIED DATASETS that exist on HuggingFace (use one of these EXACT ids with load_dataset):\n"
                f"{lst}\nAlternatively use a torchvision (e.g. CIFAR10) or sklearn built-in dataset. "
                "Load only a small subset, e.g. split='test[:500]'.\n")

    def plan_formulation(self) -> bool:
        plan = self._dialogue("plan_formulation", self.phd, self.postdoc, "PLAN",
                              f"Current Literature Review:\n{self.lit_str()}")
        self.plan = plan or ""
        return bool(plan)

    def data_preparation(self) -> bool:
        phase = "data_preparation"
        memory_context = self.mem(phase, self.plan[:500])  # [MEMÓRIA]
        context = (f"Current Literature Review:\n{self.lit_str()}\nCurrent Plan:\n{self.plan}\n"
                   f"{self.suggest_datasets()}")
        fb_mle = fb_phd = ""
        for step in range(self.max_steps):
            rm = self.mle.step(self.topic, phase, step, self.max_steps, fb_mle, context, memory_context)
            if (code := extract_command(rm, "python")) is not None:
                ok, out = execute_code(code, timeout=300)
                fb_mle = f"Code output:\n{out}"
                fb_phd = f"The ML engineer ran code. Output:\n{out[:1500]}"
            elif (q := extract_command(rm, "SEARCH_HF")) is not None:
                fb_mle = f"HF datasets:\n{search_hf_datasets(q)}"
                fb_phd = f"The ML engineer searched HF for '{q}'."
            else:
                fb_phd = extract_command(rm, "DIALOGUE") or rm[:1500]
            rp = self.phd.step(self.topic, phase, step, self.max_steps, fb_phd, context, memory_context)
            self.last_response[phase] = rp
            if (code := extract_command(rp, "SUBMIT_CODE")) is not None and code:
                ok, out = execute_code(code, timeout=300)
                if ok:
                    self.dataset_code = code
                    self.metrics["phase_steps"][phase] = step + 1
                    return True
                self.errors.append({"phase": phase, "text": out[-500:]})
                if self.memory is not None:  # [MEMÓRIA] erro na preparação de dados
                    self.memory.add(f"Erro ao preparar dados: {out.strip().splitlines()[-1][:300] if out.strip() else 'sem saída'}",
                                    "error", self.topic, phase=phase, importance=0.5)
                fb_mle = fb_phd = (f"Submitted code failed:\n{out[-1500:]}\nIf the dataset was not found or needs a "
                                   "loading script, choose ANOTHER exact ID from the verified list (or a torchvision/"
                                   "sklearn built-in dataset). Never invent dataset names.")
            else:
                fb_mle = extract_command(rp, "DIALOGUE")
                if fb_mle is None:
                    fb_mle = rp[:1500]
                    self.invalid(phase, rp, "```DIALOGUE\nmessage\n```  or  ```SUBMIT_CODE\nfinal python code\n```")
        self.metrics["phase_steps"][phase] = self.max_steps
        return False

    def running_experiments(self) -> bool:
        solver = MLESolver(self.llm, self.plan, self.dataset_code, insights=self.lit_str(),
                           notes=self.notes.get("running_experiments", ""), steps=self.mle_steps,
                           timeout=self.exp_timeout, memory=self.memory, topic=self.topic)
        self.exp_code, self.exp_results, self.mle_score = solver.solve()
        self.errors += [{"phase": "running_experiments", "text": e} for e in solver.errors]
        self.metrics["mle_solver"] = solver.stats
        return bool(self.exp_code)

    def results_interpretation(self) -> bool:
        ctx = (f"Current Literature Review:\n{self.lit_str()}\nCurrent Plan:\n{self.plan}\n"
               f"Current Dataset code:\n{self.dataset_code[:2000]}\nCurrent Experiment code:\n{self.exp_code[:3000]}\n"
               f"Current Results:\n{self.exp_results[:3000]}")
        interp = self._dialogue("results_interpretation", self.phd, self.postdoc, "INTERPRETATION", ctx)
        self.interpretation = interp or ""
        return bool(interp)

    def report_writing(self) -> bool:
        ps = PaperSolver(self.llm, self.topic, self.plan, self.lit_str(), self.exp_code, self.exp_results,
                         self.interpretation, notes=self.notes.get("report_writing", ""),
                         refine_steps=self.paper_steps, memory=self.memory)
        self.report, self.review = ps.solve()
        self.report_md = ps.markdown()
        return bool(self.report)

    def report_refinement(self) -> str:
        """PhD decide: FINISH ou voltar para um subtarefa (plan/experiment/interpretation)."""
        sys = ("You are a computer science PhD student who submitted a paper. Given the reviews, decide whether to "
               "finish or revisit an earlier stage. Reply ```FINISH\\n``` or ```REVISIT\\n<plan_formulation|"
               "running_experiments|results_interpretation>\\n```.")
        mem = self.mem("report_refinement")  # [MEMÓRIA]
        resp = self.llm.query(sys, f"{mem}Review:\n{json.dumps(self.review)[:3000]}\nPlease decide now.")
        rev = extract_command(resp, "REVISIT")
        return rev.strip() if rev else "FINISH"

    # ------------------------------------------------------------------ execução
    def _run_phase(self, phase: str) -> bool:
        # memória de curto prazo é POR FASE: sem isso o modelo imita comandos da fase anterior
        for ag in (self.phd, self.postdoc, self.mle):
            ag.reset()
        t0 = time.time()
        ok = getattr(self, phase)()
        self.metrics["phase_time"][phase] = round(time.time() - t0, 1)
        self.metrics["phase_success"][phase] = bool(ok)
        inv = self.metrics["invalid_commands"].get(phase, 0)
        self.log(f"{phase}: {'ok' if ok else 'FALHOU'} ({self.metrics['phase_steps'].get(phase, '-')} passos, "
                 f"{inv} comandos inválidos, {self.metrics['phase_time'][phase]}s)")
        if not ok and phase in self.last_response:
            self.log("   última resposta do modelo: " + self.last_response[phase][:300].replace("\n", " | "))
        errs = [e["text"] for e in self.errors if e.get("phase") == phase]
        if not ok and errs:  # o erro de execução é o que explica a falha, não só a resposta do modelo
            linhas = [l for l in errs[-1].strip().splitlines() if l.strip()]
            self.log(f"   erros de execução nesta fase: {len(errs)}; último: " + " | ".join(linhas[-3:])[:400])
        return ok

    def run(self) -> dict:
        t0 = time.time()
        if self.memory is not None:
            self.memory.start_run()
        order = PHASES[:-1]
        i = 0
        while i < len(order):
            phase = order[i]
            ok = self._run_phase(phase)
            out = {"literature_review": self.lit_str(), "plan_formulation": self.plan,
                   "data_preparation": self.dataset_code, "running_experiments": self.exp_results,
                   "results_interpretation": self.interpretation, "report_writing": self.report}[phase]
            fb = self.checkpoint(phase, out)
            if fb:  # co-pilot pediu para refazer
                if phase == "literature_review":
                    self.lit_review = []
                self.notes[phase] = (self.notes.get(phase, "") + "\n" + fb).strip()
                for ag in (self.phd, self.postdoc, self.mle):
                    ag.reset()
                continue
            if not ok and phase in ("data_preparation", "running_experiments"):
                self.log("Fase crítica falhou; encerrando o projeto.")
                break
            i += 1
            if i == len(order) and self.max_revisits > 0:
                target = self.report_refinement()
                self.metrics.setdefault("revisits", []).append(target)
                if target in order:
                    self.max_revisits -= 1
                    i = order.index(target)
        result = self.finalize(self.summary(time.time() - t0))
        if self.memory is not None:
            if not self.memory.read_only:  # [MEMÓRIA] modo online opcional: grava lições do revisor LLM
                consolidate_run(self.memory, self.llm, result, project=self.out_dir)
            result["memory_stats"] = self.memory.stats()
            result["memory_retrievals"] = self.memory.retrieval_log
        self.save(result)
        return result

    def summary(self, elapsed: float) -> dict:
        ps = self.metrics["phase_success"]
        return {
            "topic": self.topic, "with_memory": self.memory is not None,
            "review": self.review, "overall": self.review.get("Overall"),
            "mle_score": self.mle_score, "n_code_errors": len(self.errors),
            "workflow_success": round(sum(ps.values()) / max(1, len(ps)), 3),
            "usage": self.llm.usage.to_dict(), "elapsed_s": round(elapsed, 1), "metrics": self.metrics,
            "trace": {"plan": self.plan, "exp_results": self.exp_results[-3000:],
                      "interpretation": self.interpretation, "errors": self.errors[:30],
                      "n_papers": len(self.lit_review), "human_feedback": self.human_feedback},
        } | {"failures": []}

    def finalize(self, result: dict) -> dict:
        result["failures"] = detect_failures(result, self.max_steps)
        return result

    def feedback_summary(self, result: dict) -> str:
        """Tudo o que um humano precisa para dar feedback sobre o projeto (condição C), num só arquivo."""
        fails = "\n".join(f"- {f['fase']}: {f['categoria']}" for f in result.get("failures", [])) or "- nenhuma detectada"
        ok = "\n".join(f"- {k}: {'ok' if v else 'FALHOU'}" for k, v in self.metrics["phase_success"].items())
        return (f"# Projeto: {self.topic}\n\n## Etapas\n{ok}\n\n## Falhas detectadas automaticamente\n{fails}\n\n"
                f"## Artigos da revisão ({len(self.lit_review)})\n{self.lit_str() or '(nenhum)'}\n\n"
                f"## Plano\n{self.plan or '(sem plano)'}\n\n## Código de preparação de dados\n```python\n"
                f"{self.dataset_code[:3000]}\n```\n\n## Código do experimento\n```python\n{self.exp_code[:4000]}\n```\n\n"
                f"## Saída do experimento\n```\n{self.exp_results[-2500:]}\n```\n\n## Interpretação\n"
                f"{self.interpretation or '(sem interpretação)'}\n\n## Relatório final\n\n{self.report_md or '(sem relatório)'}\n")

    def save(self, result: dict):
        os.makedirs(self.out_dir, exist_ok=True)
        files = {"report.tex": self.report, "report.md": self.report_md, "resumo_para_feedback.md": self.feedback_summary(result),
                 "plan.txt": self.plan, "dataset_code.py": self.dataset_code,
                 "experiment_code.py": self.exp_code, "results.txt": self.exp_results,
                 "interpretation.txt": self.interpretation, "literature.txt": self.lit_str()}
        for name, content in files.items():
            with open(os.path.join(self.out_dir, name), "w", encoding="utf-8") as f:
                f.write(content or "")
        with open(os.path.join(self.out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
