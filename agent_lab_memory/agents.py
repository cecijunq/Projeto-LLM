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

CMD_FORMAT = ("Use exactly ONE command per turn, wrapped in triple backticks: ```COMMAND\\n<content>\\n``` "
              "where COMMAND is the actual command name.")

PHASE_TASK = {
    ("phd", "literature_review"):
        "Perform a literature review for the research topic and add relevant papers. "
        "You can search arXiv for paper summaries and read full texts. Do not overuse SUMMARY: add papers.",
    ("phd", "plan_formulation"):
        "You talk with a postdoc to design a SIMPLE experiment for the topic, building on the literature. "
        "Specify models, datasets (HuggingFace) and exact experimental steps.",
    ("postdoc", "plan_formulation"):
        "You direct a PhD student to design a SIMPLE experiment for the topic, building on the literature. "
        "When the plan is good, submit it. Do not talk for too long.",
    ("phd", "data_preparation"):
        "You direct an ML engineer to write SIMPLE code that prepares the data for the plan. "
        "When the code is ready, submit it. Use ONLY a dataset id from the verified list in the context "
        "(or a torchvision/sklearn built-in dataset); never invent dataset names. Keep the data small.",
    ("mle", "data_preparation"):
        "Write SIMPLE data preparation code for the plan, guided by the PhD student. You can run code and search HF "
        "datasets. Use ONLY dataset ids from the verified list (or torchvision/sklearn); never invent names.",
    ("phd", "results_interpretation"):
        "Discuss with the postdoc how to interpret the experimental results. Use the real numbers, metrics and significance.",
    ("postdoc", "results_interpretation"):
        "Guide the PhD student to an accurate interpretation of the results (with numbers and significance). "
        "Submit it in reasonable time.",
}

PHASE_COMMANDS = {
    ("phd", "literature_review"):
        "```SUMMARY\\n<search query>\\n``` returns top arXiv summaries.\n"
        "```FULL_TEXT\\n<arXiv id>\\n``` returns the full paper text.\n"
        "```ADD_PAPER\\n<arXiv id>\\n<summary of the paper and its results>\\n``` adds a paper to the review.",
    ("phd", "plan_formulation"): "```DIALOGUE\\n<message>\\n```",
    ("postdoc", "plan_formulation"): "```DIALOGUE\\n<message>\\n```  or  ```PLAN\\n<final plan>\\n```",
    ("phd", "data_preparation"): "```DIALOGUE\\n<message>\\n```  or  ```SUBMIT_CODE\\n<final python code>\\n```",
    ("mle", "data_preparation"):
        "```python\\n<code to run>\\n```  or  ```SEARCH_HF\\n<query>\\n```  or  ```DIALOGUE\\n<message>\\n```",
    ("phd", "results_interpretation"): "```DIALOGUE\\n<message>\\n```",
    ("postdoc", "results_interpretation"): "```DIALOGUE\\n<message>\\n```  or  ```INTERPRETATION\\n<interpretation>\\n```",
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
