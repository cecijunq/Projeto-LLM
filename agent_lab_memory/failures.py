"""
Detecção automática de falhas por etapa (métrica "falhas que se repetem" da proposta).

Cada projeto gera uma lista de eventos {"fase": ..., "categoria": ...}. Na análise, uma falha
"se repete" quando a mesma (fase, categoria) já tinha acontecido nos projetos de TREINO,
ou seja, a memória teve a chance de ensinar o agente a evitá-la e não ensinou.

Categorias (inspiradas nas falhas da Seção 5.2 do artigo):
  fase_falhou          a fase não terminou com sucesso
  formato_invalido     o modelo respondeu sem um comando válido
  busca_em_loop        a revisão bibliográfica gastou todos os passos
  dataset_inexistente  tentou carregar um dataset que não existe / exige script
  dependencia_ausente  ModuleNotFoundError / ImportError
  timeout              o código excedeu o tempo limite
  erro_de_sintaxe      SyntaxError / IndentationError
  falta_de_memoria     CUDA out of memory / MemoryError
  erro_de_execucao     KeyError, ValueError, TypeError, IndexError, AttributeError...
  outro_erro           qualquer outro erro de código
  acuracia_zero        o experimento reportou 0% de acurácia
  sem_relatorio        o projeto terminou sem relatório
"""
from __future__ import annotations

import re

_ERR_RULES = [
    ("dataset_inexistente", r"DatasetNotFoundError|doesn't exist on the Hub|Couldn't find|FileNotFoundError|"
                            r"trust_remote_code|Dataset scripts are no longer supported|EmptyDatasetError"),
    ("dependencia_ausente", r"ModuleNotFoundError|ImportError|No module named"),
    ("timeout", r"Timeout|TimeoutExpired"),
    ("erro_de_sintaxe", r"SyntaxError|IndentationError"),
    ("falta_de_memoria", r"out of memory|OutOfMemoryError|MemoryError"),
    ("erro_de_execucao", r"KeyError|ValueError|TypeError|IndexError|AttributeError|NameError|ZeroDivisionError|RuntimeError"),
]
_ACC = re.compile(r"acc(?:uracy)?\w*[^0-9\n]{0,12}?(\d+(?:\.\d+)?)\s*(%?)", re.IGNORECASE)


def classify_error(text: str) -> str:
    for cat, pat in _ERR_RULES:
        if re.search(pat, text or "", re.IGNORECASE):
            return cat
    return "outro_erro"


def has_zero_accuracy(results: str) -> bool:
    vals = [float(m.group(1)) for m in _ACC.finditer(results or "")]
    return bool(vals) and all(v == 0.0 for v in vals)


def detect_failures(result: dict, max_steps: int) -> list[dict]:
    m = result.get("metrics", {})
    tr = result.get("trace", {})
    ev: list[dict] = []
    for phase, ok in m.get("phase_success", {}).items():
        if not ok:
            ev.append({"fase": phase, "categoria": "fase_falhou"})
    for phase, n in m.get("invalid_commands", {}).items():
        if n:
            ev.append({"fase": phase, "categoria": "formato_invalido", "n": n})
    if m.get("phase_steps", {}).get("literature_review", 0) >= max_steps:
        ev.append({"fase": "literature_review", "categoria": "busca_em_loop"})
    for e in tr.get("errors", []):
        ev.append({"fase": e.get("phase", "?"), "categoria": classify_error(e.get("text", ""))})
    if tr.get("exp_results") and has_zero_accuracy(tr["exp_results"]):
        ev.append({"fase": "running_experiments", "categoria": "acuracia_zero"})
    if not result.get("review"):
        ev.append({"fase": "report_writing", "categoria": "sem_relatorio"})
    return ev
