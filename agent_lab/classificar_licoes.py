"""Classifica as lições das memórias B e C e cruza com as falhas dos projetos de teste.

  python classificar_licoes.py ../resultados_oficial      # grava memoria/classificacao_licoes.md

Cada lição recebe UMA categoria, por regras de palavras-chave aplicadas ao texto do problema na ordem abaixo
(a primeira que casa vence). A tabela completa, lição por lição, vai no fim do arquivo gerado para conferência.
As falhas dos testes são somadas sobre A, B e C juntos (não revela nada por versão).
"""
import collections
import glob
import json
import os
import re
import sys

SINTOMA = "Sintoma sem causa"
CATEGORIAS = [  # (nome, padrões) — ordem importa; um nome pode aparecer mais de uma vez (regra forte e fraca)
    ("Código e formato", r"syntax|indentation|final python code|submitted without changes|submission format|"
                         r"code quality|repeated errors|string literal|adapt the approach"),
    ("Ambiente e modelo", r"dependenc|install|librar|import|ollama|non-existent model|call the model|api validation"),
    (SINTOMA, r"did not reach|did not complete|did not produce a final report|was not completed|not completed "
              r"successfully|report was not generated|no report was generated|report was missing|did not execute any"),
    ("Desenho experimental", r"anchor|control|baseline|trivial|rule-based|fine-tuned|chain-of-thought|\bcot\b|"
                             r"single model|computational|queries|test size|noise simulation|converge|scope|premise"),
    ("Avaliação, análise e escrita", r"accura|string matching|metric|statistic|confidence interval|error bars|"
                                     r"evaluation|error responses|raw outputs|skipping rows|as if the results|"
                                     r"quantitative|discussion|citation|interpretation incorrectly|report lacked"),
    ("Dados", r"dataset|column|train-test split|pre-written|self-created|normaliz"),
    ("Desenho experimental", r"compar|condition|methodology|experimental setup|prompt design|prompt formulation|seed"),
    (SINTOMA, r"."),
]
# tipo de falha registrada pelo failures.py -> categoria de lição que a ataca
FALHA_PARA_CATEGORIA = {
    "erro_de_sintaxe": "Código e formato", "formato_invalido": "Código e formato",
    "dependencia_ausente": "Ambiente e modelo", "dataset_inexistente": "Dados",
    "erro_de_execucao": "Código e formato", "outro_erro": "Código e formato",
    "timeout": "Desenho experimental", "falta_de_memoria": "Desenho experimental",
    "acuracia_zero": "Avaliação, análise e escrita",
    "busca_em_loop": SINTOMA, "fase_falhou": SINTOMA, "sem_relatorio": SINTOMA,
}
FASES_CODIGO = ("data_preparation", "running_experiments")  # onde 80% das falhas acontecem


def recuperadas(base: str, licoes: list[dict]) -> list[str]:
    """Repete a consulta dos testes (memória somente leitura, temas de teste) nas fases de código e devolve a
    categoria de cada lição que entrou no prompt. Precisa do ambiente do projeto (sentence-transformers)."""
    from agent_lab_memory import Embedder, LongTermMemory
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "topics.json"), encoding="utf-8") as f:
        teste = json.load(f)["teste"]
    emb, out = Embedder(use_dense=True), {"B": [], "C": []}
    for v in "BC":
        for s in range(3):
            mem = LongTermMemory(os.path.join(base, "memoria", f"{v}_seed{s}.json"), embedder=emb, read_only=True)
            for t in teste:
                for fase in FASES_CODIGO:
                    for ln in mem.context_for_phase(fase, t).splitlines():
                        if m := re.search(r"Problem: (.*?) \| How to avoid: (.*)", ln):
                            out[v].append((categoria(m[1]), "final python code" in ln.lower()))
    return out
LINHA = re.compile(r"^- \*\*(?P<fase>[a-z_]+)\*\* · (?P<prob>.*?) → (?P<evitar>.*) \(importância (?P<imp>[\d.]+)\)")


def categoria(texto: str) -> str:
    t = texto.lower()
    return next(nome for nome, pat in CATEGORIAS if re.search(pat, t))


def main(base: str):
    licoes = []
    for v in "BC":
        for p in sorted(glob.glob(os.path.join(base, "memoria", f"licoes_{v}_seed*.md"))):
            for ln in open(p, encoding="utf-8"):
                if m := LINHA.match(ln.strip()):
                    licoes.append({"versao": v, "seed": p[-4], **m.groupdict(), "cat": categoria(m["prob"])})
    falhas = collections.Counter()
    for p in glob.glob(os.path.join(base, "teste", "[ABC]", "seed*", "t[0-4]", "result.json")):
        for f in json.load(open(p, encoding="utf-8")).get("failures", []):
            falhas[FALHA_PARA_CATEGORIA.get(f["categoria"], "Sintoma sem causa")] += 1
    nomes = list(dict.fromkeys(c for c, _ in CATEGORIAS))
    rec = recuperadas(base, licoes)
    por = {v: collections.Counter(l["cat"] for l in licoes if l["versao"] == v) for v in "BC"}
    tot = {v: sum(por[v].values()) for v in "BC"}
    tf = sum(falhas.values())
    pct = lambda n, d: f"{100 * n / d:.0f}%" if d else "-"
    md = ["# Classificação das lições (memórias B e C)\n",
          f"{tot['B']} lições em B e {tot['C']} em C (3 seeds cada). Falhas dos 45 projetos de teste (A, B e C "
          f"somados): {tf}, agrupadas pela categoria de lição que as atacaria.\n",
          "Colunas \"no prompt\": lições que de fato entram no prompt das fases de código (preparação de dados e "
          "experimentos), repetindo a consulta dos testes para os 5 temas × 3 seeds.\n",
          "| Categoria | Lições B | Lições C | No prompt B | No prompt C | Falhas nos testes |",
          "| --- | --- | --- | --- | --- | --- |"]
    rc = {v: collections.Counter(c for c, _ in rec[v]) for v in "BC"}
    rt = {v: len(rec[v]) for v in "BC"}
    for c in nomes:
        md.append(f"| {c} | {por['B'][c]} ({pct(por['B'][c], tot['B'])}) | {por['C'][c]} ({pct(por['C'][c], tot['C'])}) "
                  f"| {pct(rc['B'][c], rt['B'])} | {pct(rc['C'][c], rt['C'])} | {falhas[c]} ({pct(falhas[c], tf)}) |")
    alvo = [l for l in licoes if "final python code" in l["prob"].lower() + l["evitar"].lower()]
    repet = [l for l in licoes if re.search(r"same code|without changes|repeated", (l["prob"] + l["evitar"]).lower())]
    md += ["", f"- Lições que citam o marcador `</final python code>` (causa frequente de `SyntaxError`): "
               f"B = {sum(l['versao'] == 'B' for l in alvo)}, C = {sum(l['versao'] == 'C' for l in alvo)}.",
           f"- Lições sobre reenviar código que falhou ou erros repetidos: "
           f"B = {sum(l['versao'] == 'B' for l in repet)}, C = {sum(l['versao'] == 'C' for l in repet)}.",
           f"- Consultas (tema × seed × fase de código) em que a lição sobre `</final python code>` entra no prompt: "
           f"C = {sum(x for _, x in rec['C'])} de {rt['C'] // 6} (B não tem essa lição).",
           "", "## Todas as lições", "", "| Versão | Seed | Fase | Categoria | Problema |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {l['versao']} | {l['seed']} | {l['fase']} | {l['cat']} | {l['prob'].replace('|', '/')} |" for l in licoes]
    out = os.path.join(base, "memoria", "classificacao_licoes.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:12 + len(nomes)]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../resultados_oficial")
