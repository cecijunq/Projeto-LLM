"""Ferramentas: busca no arXiv, busca de datasets no HuggingFace e execução de código."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile


class ArxivSearch:
    def __init__(self, max_results: int = 20):
        self.max_results = max_results
        try:
            import arxiv
            self._client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=3)
            self._arxiv = arxiv
        except ImportError:
            self._client = None

    def summaries(self, query: str) -> str:
        if self._client is None:
            return "arXiv indisponível (pip install arxiv)."
        search = self._arxiv.Search(query=query[:300], max_results=self.max_results,
                                    sort_by=self._arxiv.SortCriterion.Relevance)
        out = []
        try:
            for r in self._client.results(search):
                out.append(f"Title: {r.title}\narXiv ID: {r.get_short_id()}\n"
                           f"Published: {r.published.date()}\nSummary: {r.summary[:800]}\n")
        except Exception as e:
            return f"Erro no arXiv: {e}"
        return "\n".join(out) or "Nenhum resultado."

    def full_text(self, arxiv_id: str, max_chars: int = 20000) -> str:
        if self._client is None:
            return "arXiv indisponível."
        try:
            from pypdf import PdfReader
            paper = next(self._client.results(self._arxiv.Search(id_list=[arxiv_id.strip()])))
            with tempfile.TemporaryDirectory() as d:
                pdf = paper.download_pdf(dirpath=d)
                text = "\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages)
            return text[:max_chars]
        except Exception as e:
            return f"Não foi possível obter o texto completo: {e}"


def search_hf_datasets(query: str, limit: int = 10) -> str:
    try:
        from huggingface_hub import list_datasets
        res = list_datasets(search=query, limit=limit, sort="downloads")
        lines = [f"- {d.id} (downloads: {getattr(d, 'downloads', '?')})" for d in res]
        return "\n".join(lines) or "Nenhum dataset encontrado."
    except Exception as e:
        return f"Busca no HuggingFace indisponível: {e}"


def describe_hf_dataset(dataset_id: str, n: int = 200) -> str | None:
    """Chamada pronta de load_dataset, com splits e colunas reais; None se o dataset não carrega
    (exige script, não tem arquivos suportados...). Só lê metadados: não baixa os dados."""
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset_builder
        configs = get_dataset_config_names(dataset_id)
        config = next((c for c in ("main", "default", "all") if c in configs), configs[0]) if configs else None
        splits = get_dataset_split_names(dataset_id, config)
        feats = load_dataset_builder(dataset_id, config).info.features
    except Exception:
        return None
    split = "test" if "test" in splits else splits[0]
    cfg = f", {config!r}" if config and config != "default" else ""
    cols = ", ".join(list(feats)[:10]) if feats else "?"
    rest = [c for c in configs if c != config]
    other = f" | other configs: {', '.join(rest[:8])}{', ...' if len(rest) > 8 else ''}" if rest else ""
    return (f"load_dataset({dataset_id!r}{cfg}, split='{split}[:{n}]')  "
            f"# splits: {', '.join(splits)} | columns: {cols}{other}")


_FORBIDDEN = [r"\bexit\s*\(\s*\)", r"\bquit\s*\(\s*\)", r"sys\.exit\s*\([^)]*\)", r"os\._exit\s*\([^)]*\)"]


def sanitize_code(code: str) -> str:
    """O artigo relata que o mle-solver gerava exit() e matava o processo; removemos isso."""
    for pat in _FORBIDDEN:
        code = re.sub(pat, "pass", code)
    return code


def execute_code(code: str, timeout: int = 600, workdir: str | None = None, max_chars: int = 4000) -> tuple[bool, str]:
    """Roda o código num subprocesso separado (isolamento mínimo).
    ATENÇÃO: código gerado por LLM deve rodar em container/VM descartável."""
    code = sanitize_code(code)
    workdir = workdir or tempfile.mkdtemp(prefix="agentlab_")
    path = os.path.join(workdir, "run.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        p = subprocess.run([sys.executable, path], cwd=workdir, capture_output=True,
                           text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        ok = p.returncode == 0
    except subprocess.TimeoutExpired:
        ok, out = False, (f"[ERRO] Timeout: the code ran for more than {timeout}s and was stopped. Make it much "
                          "faster: use a small subset of the data (e.g. at most 2000 training examples, flattened "
                          "features for sklearn) and at most 50 language model queries.")
    if len(out) > max_chars:  # evita estourar o limite de tokens (falha citada no artigo)
        out = out[: max_chars // 2] + "\n...[saída truncada]...\n" + out[-max_chars // 2:]
    return ok, out