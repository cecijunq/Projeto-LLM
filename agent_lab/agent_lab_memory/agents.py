"""
Agentes (PhD, Postdoc, ML Engineer, Professor) no estilo do Agent Laboratory.

A estrutura do prompt segue o Apêndice B.1 do artigo (system = papel + instruções da fase
+ comandos; user = contexto + histórico + passo + tópico + feedback + notas). A única
diferença estrutural é o bloco `memory_context`, inserido antes do histórico.
Os textos dos prompts são versões condensadas/adaptadas dos do Apêndice B.
"""
from __future__ import annotations

ROLE = {
    "phd": "a computer science PhD student at a top university",
    "postdoc": "a computer science postdoctoral researcher at a top university",
    "mle": "a machine learning engineer working at a top university",
    "professor": "a computer science professor at a top university",
}

CMD_FORMAT = ("Use exactly ONE of the commands above per turn, written exactly as shown: three backticks, the "
              "command name (e.g. DIALOGUE), a new line, your content (not the <placeholder>), then three backticks.")

PHASE_TASK = {
    ("phd", "literature_review"):
        "Perform a literature review for the research topic and add relevant papers. "
        "You can search arXiv for paper summaries and read full texts. Do not overuse SUMMARY: add papers.",
    ("phd", "plan_formulation"):
        "You talk with a postdoc to design a SIMPLE experiment for the topic, building on the literature. "
        "Specify the models, the data source and the exact experimental steps. Data source: a HuggingFace dataset "
        "(full 'namespace/name' id), a torchvision/sklearn built-in dataset, or a small hand-written set of 20-50 "
        "examples (often best for topics about language model behavior). If the topic is about language models, test "
        "the local language model served by Ollama (the experiment code calls it through an OpenAI-compatible API); "
        "do not download or train models with transformers, and never plan paid APIs (OpenAI, davinci, GPT-4).",
    ("postdoc", "plan_formulation"):
        "You direct a PhD student to design a SIMPLE experiment for the topic, building on the literature. "
        "The plan must run in a few minutes on one GPU: small data, and only the local language model served "
        "by Ollama (no paid APIs, no downloading models with transformers). When the plan is good, submit it. Do not talk for too long.",
    ("phd", "data_preparation"):
        "You direct an ML engineer to write SIMPLE code that prepares the data for the plan. "
        "When the code is ready, submit it as a complete script that runs on its own (include all imports). "
        "Use ONE of these data sources: (a) an EXACT dataset id from the verified HuggingFace list in the context, always the full 'namespace/name'; "
        "(b) a torchvision/sklearn built-in dataset (best for classic ML or vision topics); (c) if the topic is about "
        "how a language model behaves (biases, prompting, robustness) and no listed dataset clearly fits, a small "
        "hand-written list of 20-50 examples defined directly in the code. Never invent dataset names. Keep the data small.",
    ("mle", "data_preparation"):
        "Write SIMPLE data preparation code for the plan, guided by the PhD student. You can run code and search HF "
        "datasets. Data sources allowed: an exact id from the verified list, a torchvision/sklearn built-in dataset, "
        "or a small hand-written list of examples in the code (for topics about language model behavior). "
        "Never invent dataset names.",
    ("phd", "results_interpretation"):
        "Discuss with the postdoc how to interpret the experimental results. Use the real numbers, metrics and significance.",
    ("postdoc", "results_interpretation"):
        "Guide the PhD student to an accurate interpretation of the results (with numbers and significance). "
        "Submit it in reasonable time.",
}

PHASE_COMMANDS = {
    ("phd", "literature_review"):
        "```SUMMARY\n<search query>\n``` returns top arXiv summaries.\n"
        "```FULL_TEXT\n<arXiv id>\n``` returns the full paper text.\n"
        "```ADD_PAPER\n<arXiv id>\n<summary of the paper and its results>\n``` adds a paper to the review.",
    ("phd", "plan_formulation"): "```DIALOGUE\n<message>\n```",
    ("postdoc", "plan_formulation"): "```DIALOGUE\n<message>\n```\nor\n```PLAN\n<final plan>\n```",
    ("phd", "data_preparation"): "```DIALOGUE\n<message>\n```\nor\n```SUBMIT_CODE\n<final python code>\n```",
    ("mle", "data_preparation"):
        "```python\n<code to run>\n```\nor\n```SEARCH_HF\n<query>\n```\nor\n```DIALOGUE\n<message>\n```",
    ("phd", "results_interpretation"): "```DIALOGUE\n<message>\n```",
    ("postdoc", "results_interpretation"): "```DIALOGUE\n<message>\n```\nor\n```INTERPRETATION\n<interpretation>\n```",
}


class Agent:
    def __init__(self, kind: str, llm, max_history: int = 10, notes: dict | None = None):
        self.kind, self.llm = kind, llm
        self.max_history = max_history
        self.history: list[str] = []      # memória de CURTO prazo (como no original)
        self.prev_comm = ""
        self.notes = notes or {}           # notas do humano por fase (task notes do artigo)

    def system_prompt(self, phase: str) -> str:
        return (f"You are {ROLE[self.kind]}.\n"
                f"Task instructions: {PHASE_TASK.get((self.kind, phase), '')}\n"
                f"Commands available:\n{PHASE_COMMANDS.get((self.kind, phase), '')}\n{CMD_FORMAT}")

    def step(self, topic: str, phase: str, step: int, max_steps: int, feedback: str = "",
             context: str = "", memory_context: str = "") -> str:
        complete = ("You must finish this task and submit as soon as possible!"
                    if step >= 0.7 * max_steps else "")
        hist = "\n".join(self.history[-self.max_history:])
        prompt = (f"{context}\n{memory_context}\nHistory:\n{hist}\n"
                  f"Current Step #{step}, Phase: {phase}\n{complete}\n"
                  f"[Objective] Your goal is to perform research on the following topic: {topic}\n"
                  f"Feedback: {feedback}\nNotes: {self.notes.get(phase, '')}\n"
                  f"Your previous command was: {self.prev_comm[:300]}. Make sure your new output is different.\n"
                  "Please produce a single command below:")
        resp = self.llm.query(self.system_prompt(phase), prompt)
        self.history.append(f"Step #{step}, Phase: {phase}, Feedback: {feedback[:800]}, Your response: {resp[:800]}")
        self.prev_comm = resp
        return resp

    def reset(self):
        self.history, self.prev_comm = [], ""