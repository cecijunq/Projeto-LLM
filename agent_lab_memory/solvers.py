"""
mle-solver, paper-solver e revisor automático (versões compactas das Seções 3.2 e 3.3).
Pontos de integração com a memória estão marcados com  # [MEMÓRIA].
"""
from __future__ import annotations

import json
import random
import re

from .llm import extract_command
from .tools import execute_code

# ============================================================================ mle-solver
MLE_ROLE = ("You are an expert machine learning engineer working at a top university to write code "
            "to solve machine learning research challenges.")
MLE_PHASE = ("You are an ML engineer and you will be writing the code for a research project. Produce SIMPLE code "
             "that runs every experiment in the plan and prints the results. The dataset code is always prepended "
             "to your code, so do not rewrite it. Do not write a main function. Use only pytorch or scikit-learn "
             "(no tensorflow/keras). Before printing each result, print what it is meant to show. "
             "Your method must not get 0% accuracy; if it does, something is wrong. Keep runtime short.\n"
             "If the experiment needs a language model, call the local model exactly like this "
             "(and evaluate at most 50 examples):\n"
             "import os\nfrom openai import OpenAI\n"
             "client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY', 'x'))\n"
             "r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], "
             "messages=[{'role': 'user', 'content': prompt}], max_tokens=64, temperature=0)\n"
             "answer = r.choices[0].message.content")
MLE_CMDS = ("Commands (ONE per turn):\n"
            "```REPLACE\\n<entire new python file>\\n``` rewrites all code (use for big changes).\n"
            "```EDIT N M\\n<new lines>\\n``` replaces lines N..M (inclusive) of the current code.\n"
            "Code is tested before being accepted.")
SCORER_SYS = ("You are a professor agent serving as an expert reward model: read a research plan, research code and "
              "its output, and score from 0 to 1 how well the code follows the plan and produces proper results. "
              "Answer exactly as ```SCORE\\n<float>\\n```")
REPAIR_SYS = ("You are an automated code repair tool. Fix the code so the error does not happen again, changing as "
              "little as possible. Return the full code inside ```python ... ```.")


class MLESolver:
    def __init__(self, llm, plan: str, dataset_code: str, insights: str = "", notes: str = "",
                 steps: int = 3, top_k: int = 2, n_repair: int = 2, timeout: int = 600,
                 memory=None, topic: str = ""):
        self.llm, self.plan, self.dataset_code = llm, plan, dataset_code
        self.insights, self.notes = insights, notes
        self.steps, self.top_k, self.n_repair, self.timeout = steps, top_k, n_repair, timeout
        self.memory, self.topic = memory, topic
        self.top: list[tuple[float, list[str], str]] = []  # (score, linhas, saída)
        self.reflection = ""
        self.errors: list[str] = []          # usado na consolidação da memória
        self.stats = {"exec_attempts": 0, "exec_errors": 0, "repairs_ok": 0}

    # ---- memória: erros parecidos e lições relevantes
    def _memory_block(self, query: str) -> str:
        if self.memory is None:
            return ""
        return self.memory.context_for_phase("running_experiments", self.topic, extra_query=query, k=6)  # [MEMÓRIA]

    def _system(self, code_lines: list[str], last_output: str) -> str:
        code = "\n".join(f"{i} |{l}" for i, l in enumerate(code_lines)) or "(empty)"
        return (f"{MLE_ROLE}\nTask instructions: {MLE_PHASE}\n"
                f"Insights from the literature review:\n{self.insights[:3000]}\n"
                f"{self._memory_block(self.plan[:500] + last_output[-500:])}"
                f"Reflection on previous attempts:\n{self.reflection}\n"
                f"Notes: {self.notes}\nResearch plan:\n{self.plan}\n"
                f"Dataset code (prepended automatically):\n{self.dataset_code[:3000]}\n{MLE_CMDS}\n"
                f"Current code:\n{code}")

    def _run(self, lines: list[str]) -> tuple[bool, str]:
        self.stats["exec_attempts"] += 1
        ok, out = execute_code(self.dataset_code + "\n" + "\n".join(lines), timeout=self.timeout)
        if not ok:
            self.stats["exec_errors"] += 1
        return ok, out

    def _repair(self, lines: list[str], error: str) -> tuple[bool, list[str], str]:
        for _ in range(self.n_repair):
            mem = self._memory_block(error[-800:])  # [MEMÓRIA] o reparo consulta erros já vistos
            resp = self.llm.query(REPAIR_SYS + "\n" + mem, f"Error:\n{error[-2000:]}\n\nCode:\n" + "\n".join(lines),
                                  temperature=0.8)
            fixed = extract_command(resp, "python")
            if fixed is None:
                continue
            new_lines = fixed.split("\n")
            ok, out = self._run(new_lines)
            if ok:
                self.stats["repairs_ok"] += 1
                if self.memory is not None:  # [MEMÓRIA] grava erro + solução
                    self.memory.add(f"Erro: {error.strip().splitlines()[-1][:300]} -> corrigido por reparo automático.",
                                    "error", self.topic, phase="running_experiments", importance=0.5)
                return True, new_lines, out
            error = out
        return False, lines, error

    def _score(self, lines: list[str], output: str) -> float:
        resp = self.llm.query(SCORER_SYS, f"Plan:\n{self.plan}\n\nCode:\n" + "\n".join(lines) +
                              f"\n\nOutput:\n{output[-3000:]}", temperature=0.6)
        body = extract_command(resp, "SCORE") or resp
        m = re.search(r"[01](?:\.\d+)?", body)
        return float(m.group(0)) if m else 0.0

    def _apply(self, resp: str, lines: list[str]) -> list[str] | None:
        rep = extract_command(resp, "REPLACE")
        if rep is not None:
            return rep.split("\n")
        ed = extract_command(resp, "EDIT")
        if ed is not None:
            header, _, body = ed.partition("\n")
            nums = re.findall(r"\d+", header)
            if len(nums) >= 2:
                n, m = int(nums[0]), int(nums[1])
                return lines[:n] + body.split("\n") + lines[m + 1:]
        return None

    def solve(self) -> tuple[str, str, float]:
        lines: list[str] = []
        last_out = ""
        # passo inicial: gerar código do zero (até 5 tentativas)
        for _ in range(5):
            resp = self.llm.query(self._system([], ""), "Use ```REPLACE to create the initial code now.", temperature=1.0)
            new = self._apply(resp, [])
            if new is None:
                continue
            ok, out = self._run(new)
            if not ok:
                self.errors.append(out[-500:])
                ok, new, out = self._repair(new, out)
            if ok:
                self.top.append((self._score(new, out), new, out))
                break
        for step in range(self.steps):
            if not self.top:
                break
            _, base, last_out = random.choice(self.top)  # amostragem do pool de melhores programas
            resp = self.llm.query(self._system(base, last_out), f"Step {step}: improve the code with one command.",
                                  temperature=1.0)
            new = self._apply(resp, base)
            if new is None:
                continue
            ok, out = self._run(new)
            if not ok:
                self.errors.append(out[-500:])
                ok, new, out = self._repair(new, out)
            if ok:
                sc = self._score(new, out)
                self.top.append((sc, new, out))
                self.top = sorted(self.top, key=lambda t: t[0], reverse=True)[: self.top_k]
                msg = f"Code ran. Score {sc:.2f}. Output:\n{out[-1500:]}\nReflect on how to improve this result."
            else:
                msg = f"Code failed with error:\n{out[-1500:]}\nReflect on why and how to fix it."
            self.reflection = self.llm.query("You are an ML engineer. Reflect briefly and concretely.", msg, temperature=0.8)
        if not self.top:
            return "", "Nenhum código executou com sucesso.", 0.0
        best = max(self.top, key=lambda t: t[0])
        return "\n".join(best[1]), best[2], best[0]


# ============================================================================ revisor
REVIEW_SYS = (
    "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. "
    "Be critical and cautious. Respond with 'THOUGHT:' (your reasoning) and then 'REVIEW JSON:' with a ```json``` block "
    "containing: Summary, Strengths (list), Weaknesses (list), Originality, Quality, Clarity, Significance (1-4), "
    "Questions, Limitations, Ethical Concerns (bool), Soundness, Presentation, Contribution (1-4), "
    "Overall (1-10, NeurIPS scale: 6 weak accept, 5 borderline accept, 4 borderline reject, 3 reject), "
    "Confidence (1-5), Decision (Accept or Reject). Deduct points if any core section is missing."
)


def review_paper(llm, plan: str, latex: str) -> dict:
    resp = llm.query(REVIEW_SYS, f"Research plan:\n{plan}\n\nPaper:\n{latex[:20000]}", temperature=0.5)
    body = extract_command(resp, "json") or resp
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", body, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


# ============================================================================ paper-solver
SECTIONS = ["Abstract", "Introduction", "Background", "Related Work", "Methods",
            "Experimental Setup", "Results", "Discussion"]
TIPS = {
    "Abstract": "One paragraph: what, why it matters, why hard, contribution, how verified.",
    "Related Work": "Compare and contrast with alternative approaches; do not just describe them.",
    "Methods": "Precise methodology and equations.",
    "Experimental Setup": "Dataset, metrics, hyperparameters. Do not invent hardware details.",
    "Results": "ONLY results that were actually run. Numbers, baselines, limitations. Never hallucinate results.",
}
PAPER_SYS = ("You are a computer science PhD student acting as a report writer: you summarize research that was "
             "actually performed into an academic report. Writing style: objective. Write LaTeX body text only "
             "(no packages, no \\section commands). Return it as ```REPLACE\\n<text>\\n```.")


class PaperSolver:
    def __init__(self, llm, topic: str, plan: str, lit_review: str, exp_code: str, exp_results: str,
                 interpretation: str, notes: str = "", refine_steps: int = 2, memory=None):
        self.llm, self.topic, self.plan = llm, topic, plan
        self.lit, self.code, self.results, self.interp = lit_review, exp_code, exp_results, interpretation
        self.notes, self.refine_steps, self.memory = notes, refine_steps, memory
        self.sections: dict[str, str] = {}

    def _context(self) -> str:
        mem = self.memory.context_for_phase("report_writing", self.topic, k=5) if self.memory else ""  # [MEMÓRIA]
        return (f"{mem}Topic: {self.topic}\nNotes: {self.notes}\nLiterature review:\n{self.lit[:3000]}\n"
                f"Plan:\n{self.plan[:2000]}\nCode:\n{self.code[:3000]}\nResults:\n{self.results[:2000]}\n"
                f"Interpretation:\n{self.interp[:1500]}\n")

    def latex(self) -> str:
        body = "\n".join(f"\\section{{{s}}}\n{self.sections.get(s, '')}" for s in SECTIONS)
        return ("\\documentclass{article}\n\\usepackage{amsmath}\n\\title{Research Report: " + self.topic +
                "}\n\\author{Agent Laboratory}\n\\begin{document}\n\\maketitle\n" + body + "\n\\end{document}")

    def markdown(self) -> str:
        """Versão legível do relatório (para leitura e avaliação humana)."""
        body = "\n\n".join(f"## {s}\n\n{self.sections.get(s, '')}" for s in SECTIONS)
        return f"# Research Report: {self.topic}\n\n{body}\n"

    def write_section(self, s: str, extra: str = "") -> None:
        resp = self.llm.query(PAPER_SYS, f"{self._context()}\nCurrent paper sections so far: "
                              f"{', '.join(self.sections)}\n{extra}\nWrite the section: {s}. Tips: {TIPS.get(s, '')}",
                              temperature=0.8)
        text = extract_command(resp, "REPLACE")
        self.sections[s] = text if text is not None else resp

    def solve(self) -> tuple[str, dict]:
        for s in SECTIONS:
            self.write_section(s)
        best_tex = self.latex()
        best_rev = review_paper(self.llm, self.plan, best_tex)
        for _ in range(self.refine_steps):  # edição guiada pela revisão; mantém a melhor versão
            weak = "; ".join(best_rev.get("Weaknesses", [])[:4])
            target = random.choice(["Introduction", "Methods", "Results", "Discussion"])
            old = self.sections[target]
            self.write_section(target, extra=f"Reviewer weaknesses to address: {weak}\nPrevious version:\n{old[:3000]}")
            tex = self.latex()
            rev = review_paper(self.llm, self.plan, tex)
            if rev.get("Overall", 0) >= best_rev.get("Overall", 0):
                best_tex, best_rev = tex, rev
            else:
                self.sections[target] = old
        return best_tex, best_rev
