"""
Camada de acesso a LLMs.

- LLM: wrapper para APIs compatíveis com OpenAI (OpenAI, vLLM, Ollama, OpenRouter...)
  ou para a API da Anthropic. Contabiliza tokens e chamadas (útil para comparar
  custo com e sem memória).
- MockLLM: respostas determinísticas, só para testar o pipeline sem gastar API.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class LLM:
    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai",
                 temperature: float = 0.8, max_tokens: int = 4096,
                 base_url: str | None = None, api_key: str | None = None, seed: int | None = None):
        self.model, self.provider = model, provider
        self.seed = seed  # reduz a variância entre as versões A/B/C (OpenAI e Ollama aceitam 'seed')
        self.temperature, self.max_tokens = temperature, max_tokens
        self.usage = Usage()
        if provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY"),
                                  timeout=180)  # chamada travada no Ollama não prende a execução (query tenta 4x)
        elif provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"provider desconhecido: {provider}")

    def query(self, system: str, prompt: str, temperature: float | None = None) -> str:
        temp = self.temperature if temperature is None else temperature
        last_err = None
        for attempt in range(4):
            try:
                if self.provider == "openai":
                    extra = {"seed": self.seed} if self.seed is not None else {}
                    r = self._client.chat.completions.create(
                        model=self.model, temperature=temp, max_tokens=self.max_tokens, **extra,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": prompt}])
                    text = r.choices[0].message.content or ""
                    if r.usage:
                        self.usage.prompt_tokens += r.usage.prompt_tokens
                        self.usage.completion_tokens += r.usage.completion_tokens
                else:
                    r = self._client.messages.create(
                        model=self.model, temperature=temp, max_tokens=self.max_tokens,
                        system=system, messages=[{"role": "user", "content": prompt}])
                    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    self.usage.prompt_tokens += r.usage.input_tokens
                    self.usage.completion_tokens += r.usage.output_tokens
                self.usage.calls += 1
                log_call(self, system, prompt, text)
                return text
            except Exception as e:  # rate limit, timeout...
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Falha ao consultar o LLM: {last_err}")


class MockLLM:
    """LLM falso: devolve comandos válidos para cada fase. Serve só para testar o
    fluxo (memória, persistência, métricas) sem custo. Os 'resultados' são aleatórios."""

    def __init__(self, seed: int = 0):
        self.usage = Usage()
        self.rng = random.Random(seed)

    def query(self, system: str, prompt: str, temperature: float | None = None) -> str:
        self.usage.calls += 1
        self.usage.prompt_tokens += (len(system) + len(prompt)) // 4
        text = self._answer(system, prompt)
        self.usage.completion_tokens += len(text) // 4
        log_call(self, system, prompt, text)
        return text

    def _answer(self, system: str, prompt: str) -> str:
        cur = re.findall(r"Current Step #\d+, Phase: (\w+)", prompt)  # fase ATUAL (não a do histórico)
        s = system + "\n" + (f"Phase: {cur[-1]}" if cur else prompt)
        bonus = 0.0  # mock neutro: não simula ganho com memória (isso é o que o experimento real mede)
        if "expert reward model" in system:
            return f"```SCORE\n{min(1.0, self.rng.uniform(0.4, 0.8) + bonus):.2f}\n```"
        if "reviewing a paper" in system:
            o = min(10, self.rng.randint(3, 6) + (1 if bonus else 0))
            return "THOUGHT: ok\nREVIEW JSON:\n```json\n" + json.dumps({
                "Summary": "mock", "Strengths": ["clear"], "Weaknesses": ["small dataset", "no significance test"],
                "Originality": 2, "Quality": 2, "Clarity": 3, "Significance": 2, "Soundness": 2,
                "Presentation": 3, "Contribution": 2, "Overall": o, "Confidence": 3,
                "Decision": "Reject" if o < 6 else "Accept"}) + "\n```"
        if "extract reusable lessons" in system:
            src = "human" if "human researchers" in prompt else "llm"
            return "```json\n" + json.dumps([
                {"problem": f"[{src}] No statistical significance test for the accuracy gap.",
                 "how_to_avoid": "Report a bootstrap confidence interval for every comparison.",
                 "phase": "running_experiments", "importance": 0.8},
                {"problem": f"[{src}] Dataset too large for the timeout.",
                 "how_to_avoid": "Load a small subset (e.g. split='test[:500]').",
                 "phase": "data_preparation", "importance": 0.6}]) + "\n```"
        if "code repair tool" in system:
            return "```python\nprint('accuracy: 0.70')\n```"
        if "Reflect" in prompt or "reflect" in prompt[:200]:
            return "Use a stronger baseline and print per-class accuracy."
        if "Phase: literature_review" in s:
            if "Paper summaries" in prompt or self.rng.random() < 0.5:
                return f"```ADD_PAPER\n2401.0{self.rng.randint(1000, 9999)}\nMock paper relevant to the topic.\n```"
            return "```SUMMARY\nlanguage model bias benchmark\n```"
        if "Phase: plan_formulation" in s:
            if "postdoctoral" in system:
                return ("```PLAN\nData: a small HuggingFace dataset (stanfordnlp/sst2, split='train[:500]').\n"
                        "Models: logistic regression baseline vs. the proposed method.\n"
                        "Comparison: accuracy of the method against the baseline on the same split.\n"
                        "Metric: accuracy with a bootstrap confidence interval.\n"
                        "Steps: 1. Load the data. 2. Train the baseline. 3. Train the method. 4. Report accuracy.\n```")
            return "```DIALOGUE\nI propose comparing a baseline with our method on a small dataset.\n```"
        if "Phase: data_preparation" in s:
            if "PhD student" in system:
                return ("```SUBMIT_CODE\nimport random\nrandom.seed(0)\n"
                        "data = [(random.random(), random.randint(0, 1)) for _ in range(200)]\n"
                        "print('dataset size', len(data))\n```")
            return "```DIALOGUE\nI will generate a small toy dataset.\n```"
        if "Phase: running_experiments" in s or "ML engineer and you will be writing the code" in s:
            acc = min(0.99, self.rng.uniform(0.55, 0.8) + bonus)
            return (f"```REPLACE\nimport random\nacc = {acc:.3f}\n"
                    "print('Results show baseline vs method accuracy')\n"
                    f"print('accuracy:', acc)\n```")
        if "Phase: results_interpretation" in s:
            if "postdoctoral" in system:
                return "```INTERPRETATION\nThe method improved accuracy by ~5 points over the baseline.\n```"
            return "```DIALOGUE\nThe method seems better than the baseline.\n```"
        if "report writer" in system or "section" in prompt.lower():
            return "```REPLACE\nMock section text with numbers: accuracy 0.70.\n```"
        if "decide" in prompt.lower():
            return "```FINISH\n```"
        return "```DIALOGUE\nok\n```"


def log_call(llm, system: str, prompt: str, response: str) -> None:
    """Grava cada chamada em JSONL (se llm.log_path estiver definido) para diagnóstico."""
    path = getattr(llm, "log_path", None)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 1), "system": system[:400], "prompt_tail": prompt[-600:],
                                "response": response}, ensure_ascii=False) + "\n")
    except OSError:
        pass


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PLACEHOLDER = re.compile(r"<[a-z][a-z0-9 _/-]*>")  # placeholders dos exemplos: <message>, <final python code>...


def extract_command(text: str, command: str) -> str | None:
    """Extrai o conteúdo de um bloco ```COMMAND ... ``` (formato usado no artigo).

    Modelos menores raramente seguem o formato à risca, então aceitamos variações:
      ```PLAN\n...```            (formato canônico)
      ```plan\n...```            (minúsculas)
      ```PLAN: texto```          (tudo numa linha, com ou sem ':')
      ```text\nPLAN\n...```      (tag em minúsculas antes; comando em MAIÚSCULAS)
      ```PLAN\n...  (sem fechar) (resposta cortada: lê até o fim)
      PLAN\n...               (sem crases, no início da resposta)
    Se o conteúdo vier numa linha só com '\\n' escrito por extenso (barra + n), ele vira quebra de linha:
    modelos pequenos às vezes imitam o '\\n' em vez de quebrar a linha, e o código sai com SyntaxError.
    Blocos <think>...</think> de modelos de raciocínio são ignorados.
    """
    text = _THINK.sub("", text or "")
    cmd = re.escape(command)
    tail = r"(?![A-Za-z0-9_])[ \t]*:?[ \t]*([^\n]*?)[ \t]*(?:\n(.*?))?(?:```|\Z)"
    patterns = [re.compile(r"```[ \t]*" + cmd + tail, re.DOTALL | re.IGNORECASE)]
    if command.lower() != "python":
        patterns.append(re.compile(r"```[ \t]*[a-z]*[ \t]*\n[ \t]*" + cmd + tail, re.DOTALL))
        patterns.append(re.compile(r"\A\s*" + cmd + tail, re.DOTALL))  # sem crases (sensível a maiúsculas)
    for pat in patterns:
        m = pat.search(text)
        if m:
            header = (m.group(1) or "").strip()
            body = (m.group(2) or "").strip()
            content = "\n".join(x for x in (header, body) if x)
            if "\n" not in content and "\\n" in content:
                content = content.replace("\\n", "\n").strip()
            first, _, rest = content.partition("\n")
            if _PLACEHOLDER.fullmatch(first.strip()):  # copiou o '<final python code>' do exemplo de comando
                content = rest.strip()
            return content
    return None