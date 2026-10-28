"""
Protocolo experimental do projeto (Grupo 1 · LLM 2026/2)

  QP1. Guardar lições de projetos anteriores faz o agente se sair melhor nos próximos?
  QP2. Importa de onde vem a lição: do revisor automático ou de um humano?

Versões do agente
  A  sem memória (Agent Laboratory original)
  B  memória com lições tiradas do REVISOR LLM
  C  memória com lições tiradas do FEEDBACK HUMANO

Etapas (cada uma é um subcomando; todas podem ser retomadas se a sessão cair):

  python projeto.py treino            # roda os 5 temas de treino UMA vez (sem memória)
  python projeto.py memoria --fonte llm      # constrói a memória B a partir das revisões LLM
  python projeto.py teste --versao A         # roda os 5 temas de teste sem memória
  python projeto.py teste --versao B         # ... com a memória B congelada
  (o grupo preenche feedback_humano.csv lendo os projetos de treino)
  python projeto.py memoria --fonte humano   # constrói a memória C a partir do feedback humano
  python projeto.py teste --versao C
  python projeto.py avaliacao-cega            # gera os relatórios anonimizados + planilha de notas
  (cada avaliador preenche a sua cópia da planilha)
  python projeto.py analise                   # tabelas, gráficos e testes estatísticos

Decisões de desenho
  - B e C são construídas a partir dos MESMOS projetos de treino e do MESMO extrator de lições;
    a única diferença é o texto do feedback (revisor LLM vs. humano). Isso isola a QP2.
  - Na fase de teste a memória é somente-leitura: os 5 projetos de teste de A, B e C partem
    exatamente do mesmo estado, e a ordem dos temas não interfere.
  - Mesmo modelo e mesma seed de geração por tema nas três versões.

Use --mock em qualquer etapa para testar o fluxo sem LLM (feedback e notas humanas simulados).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys

from agent_lab_memory import AgentLaboratory, LLM, MockLLM, LongTermMemory, Embedder
from agent_lab_memory.memory import extract_lessons, format_llm_review, project_facts, store_lessons

VERSOES = {"A": "Sem memória", "B": "Memória + revisor LLM", "C": "Memória + humano"}
CRITERIOS = ["qualidade_experimental", "qualidade_relatorio", "utilidade"]


# ============================================================================ utilidades
def load_topics(path: str) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["treino"], d["teste"]


def make_llm(args, seed: int):
    if args.mock:
        return MockLLM(seed=seed)
    return LLM(model=args.model, provider=args.provider, base_url=args.base_url, seed=seed)


def read_csv(path: str) -> list[dict]:
    """Lê CSV separado por ';' ou ',' (Excel em português usa ';', Google Sheets costuma usar ',')."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters=";,").delimiter
        except csv.Error:
            delim = ";"
        return list(csv.DictReader(f, delimiter=delim))


def to_md(df) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # 'tabulate' ausente
        return "```\n" + df.to_string() + "\n```"


def write_csv(path: str, rows: list[dict], fields: list[str]):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        w.writerows(rows)


def project_dirs(base: str, seeds: int, n: int):
    for s in range(seeds):
        for i in range(n):
            yield s, i, os.path.join(base, f"seed{s}", f"t{i}")


def load_result(d: str) -> dict | None:
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_projects(args, topics: list[str], base: str, memory_for_seed, seed_offset: int):
    for s in range(args.seeds):
        memory = memory_for_seed(s)
        for i, topic in enumerate(topics):
            d = os.path.join(base, f"seed{s}", f"t{i}")
            if load_result(d) is not None:
                print(f"[pulando] {d} (já concluído)")
                continue
            print(f"\n[rodando] {d}: {topic[:70]}", flush=True)
            project_seed = seed_offset + s * 100 + i
            random.seed(project_seed)
            lab = AgentLaboratory(topic, make_llm(args, seed_offset + s * 100 + i), memory=memory,
                                  num_papers=args.num_papers, max_steps=args.max_steps, mle_steps=args.mle_steps,
                                  out_dir=d, verbose=args.verbose)
            r = lab.run()
            u = r["usage"]
            print(f"[ok] nota LLM={r['overall']} | falhas={len(r['failures'])} | "
                  f"tokens={u['prompt_tokens'] + u['completion_tokens']} | {r['elapsed_s']}s", flush=True)


# ============================================================================ etapas
FEEDBACK_FIELDS = ["projeto", "tema", "arquivo_para_ler", "nota_1a5", "o_que_deu_errado", "como_evitar"]


def cmd_treino(args):
    treino, _ = load_topics(args.topics)
    run_projects(args, treino, os.path.join(args.out, "treino"), lambda s: None, seed_offset=0)
    # planilha de feedback humano (condição C): preserva o que já foi preenchido
    path = os.path.join(args.out, "feedback_humano.csv")
    old = {r["projeto"]: r for r in read_csv(path)} if os.path.exists(path) else {}
    rows = []
    for s, i, d in project_dirs(os.path.join(args.out, "treino"), args.seeds, len(treino)):
        pid = f"seed{s}/t{i}"
        rows.append(old.get(pid) or {"projeto": pid, "tema": treino[i],
                                     "arquivo_para_ler": os.path.relpath(os.path.join(d, "resumo_para_feedback.md"), args.out),
                                     "nota_1a5": "", "o_que_deu_errado": "", "como_evitar": ""})
    write_csv(path, rows, FEEDBACK_FIELDS)
    print(f"\nPlanilha de feedback humano: {path}")


def cmd_memoria(args):
    treino, _ = load_topics(args.topics)
    letra = {"llm": "B", "humano": "C"}[args.fonte]
    fonte = {"llm": "llm", "humano": "human"}[args.fonte]
    fb_rows = {}
    if args.fonte == "humano":
        path = os.path.join(args.out, "feedback_humano.csv")
        if not os.path.exists(path):
            sys.exit("Rode a etapa 'treino' antes: ela cria feedback_humano.csv.")
        fb_rows = {r["projeto"]: r for r in read_csv(path)}
        faltando = [p for p, r in fb_rows.items() if not (r.get("o_que_deu_errado", "").strip() or r.get("como_evitar", "").strip())]
        if faltando and not args.mock:
            sys.exit(f"Feedback humano vazio para: {', '.join(faltando)}. Preencha feedback_humano.csv e rode de novo.")
    os.makedirs(os.path.join(args.out, "memoria"), exist_ok=True)
    for s in range(args.seeds):
        mem_path = os.path.join(args.out, "memoria", f"{letra}_seed{s}.json")
        if os.path.exists(mem_path):
            os.remove(mem_path)  # reconstrói do zero (a etapa é determinística dado o feedback)
        mem = LongTermMemory(mem_path, embedder=Embedder(use_dense=not args.mock))
        mem.start_run()
        llm = make_llm(args, 777_000 + s)  # mesma seed para B e C: o extrator se comporta igual
        listagem = [f"# Memória {letra} ({VERSOES[letra]}) · seed {s}\n"]
        for i, topic in enumerate(treino):
            d = os.path.join(args.out, "treino", f"seed{s}", f"t{i}")
            res = load_result(d)
            if res is None:
                sys.exit(f"Projeto de treino ausente: {d}. Rode a etapa 'treino' antes.")
            pid = f"seed{s}/t{i}"
            if fonte == "llm":
                feedback = format_llm_review(res.get("review"))
            else:
                r = fb_rows.get(pid, {})
                if args.mock and not (r.get("o_que_deu_errado") or r.get("como_evitar")):
                    r = {"nota_1a5": "2", "o_que_deu_errado": "O experimento usou poucos exemplos e não reportou "
                         "intervalo de confiança.", "como_evitar": "Usar ao menos 200 exemplos e bootstrap."}
                feedback = (f"Score given by the humans (1-5): {r.get('nota_1a5', '?')}\n"
                            f"What went wrong: {r.get('o_que_deu_errado', '')}\nHow to avoid it: {r.get('como_evitar', '')}")
            lessons = extract_lessons(llm, topic, project_facts(res), feedback, fonte)
            n = store_lessons(mem, lessons, topic, fonte, pid)
            listagem.append(f"\n## {pid} · {topic}\n" + "\n".join(
                f"- **{l['phase']}** · {l['problem']} → {l['how_to_avoid']} (importância {l['importance']:.1f})"
                for l in lessons) + (f"\n_({n} novas; o resto foi fundido com lições parecidas)_" if n < len(lessons) else ""))
        mem.save()
        with open(os.path.join(args.out, "memoria", f"licoes_{letra}_seed{s}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(listagem) + "\n")
        print(f"Memória {letra} seed {s}: {mem.stats()['insight']} lições -> {mem_path}")


def cmd_teste(args):
    _, teste = load_topics(args.topics)
    v = args.versao

    def memory_for_seed(s):
        if v == "A":
            return None
        p = os.path.join(args.out, "memoria", f"{v}_seed{s}.json")
        if not os.path.exists(p):
            fonte = "llm" if v == "B" else "humano"
            sys.exit(f"Memória {v} não encontrada ({p}). Rode: python projeto.py memoria --fonte {fonte}")
        return LongTermMemory(p, embedder=Embedder(use_dense=not args.mock), read_only=True)

    # mesma seed de geração por (seed, tema) em A, B e C
    run_projects(args, teste, os.path.join(args.out, "teste", v), memory_for_seed, seed_offset=500_000)


def cmd_avaliacao(args):
    _, teste = load_topics(args.topics)
    base = os.path.join(args.out, "avaliacao_cega")
    chave_path = os.path.join(base, ".chave.json")
    if os.path.exists(chave_path) and not args.refazer:
        sys.exit(f"A avaliação cega já foi gerada ({base}). Use --refazer para gerar de novo (embaralha outra vez).")
    itens = []
    for v in VERSOES:
        for s, i, d in project_dirs(os.path.join(args.out, "teste", v), args.seeds, len(teste)):
            if load_result(d) is not None:
                itens.append({"versao": v, "seed": s, "t": i, "dir": d, "tema": teste[i]})
    if not itens:
        sys.exit("Nenhum projeto de teste concluído ainda.")
    random.Random(2026).shuffle(itens)
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(os.path.join(base, "relatorios"))
    chave, planilha = {}, []
    for n, it in enumerate(itens, 1):
        rid = f"R{n:02d}"
        src = os.path.join(it["dir"], "report.md")
        dst = os.path.join(base, "relatorios", f"{rid}.md")
        if os.path.exists(src) and os.path.getsize(src) > 0:
            shutil.copy(src, dst)
        else:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(f"# {it['tema']}\n\n(O agente não chegou a gerar um relatório neste projeto.)\n")
        chave[rid] = it
        planilha.append({"id": rid, "arquivo": f"relatorios/{rid}.md", "tema": it["tema"],
                         **{c: "" for c in CRITERIOS}, "comentarios": ""})
    with open(chave_path, "w", encoding="utf-8") as f:
        json.dump(chave, f, ensure_ascii=False, indent=2)
    fields = ["id", "arquivo", "tema", *CRITERIOS, "comentarios"]
    write_csv(os.path.join(base, "planilha_modelo.csv"), planilha, fields)
    print(f"{len(itens)} relatórios anonimizados em {base}/relatorios")
    print("Cada avaliador copia planilha_modelo.csv como avaliacao_<seu_nome>.csv (na mesma pasta) e dá notas 1–5.")
    print("NÃO abra .chave.json antes de terminar: ela revela a versão de cada relatório.")
    if args.mock:  # avaliadores simulados, só para testar a análise
        for nome in ("mock1", "mock2"):
            rng = random.Random(nome)
            rows = [{**r, **{c: rng.randint(1, 5) for c in CRITERIOS}} for r in planilha]
            write_csv(os.path.join(base, f"avaliacao_{nome}.csv"), rows, fields)


def cmd_analise(args):
    import pandas as pd
    treino, teste = load_topics(args.topics)
    out = os.path.join(args.out, "analise")
    os.makedirs(out, exist_ok=True)

    # falhas que já ocorreram no treino (a memória "deveria" ter ensinado a evitá-las)
    vistas = set()
    for s, i, d in project_dirs(os.path.join(args.out, "treino"), args.seeds, len(treino)):
        r = load_result(d)
        for f in (r or {}).get("failures", []):
            vistas.add((f["fase"], f["categoria"]))

    rows, falhas = [], []
    for v in VERSOES:
        for s, i, d in project_dirs(os.path.join(args.out, "teste", v), args.seeds, len(teste)):
            r = load_result(d)
            if r is None:
                continue
            fs = r.get("failures", [])
            for f in fs:
                falhas.append({"versao": v, "seed": s, "t": i, "fase": f["fase"], "categoria": f["categoria"],
                               "repetida": (f["fase"], f["categoria"]) in vistas})
            u = r["usage"]
            rows.append({"versao": v, "seed": s, "t": i, "tema": teste[i][:60], "nota_llm": r.get("overall"),
                         "mle_score": r.get("mle_score"), "sucesso_etapas": r.get("workflow_success"),
                         "falhas": len(fs), "falhas_repetidas": sum((f["fase"], f["categoria"]) in vistas for f in fs),
                         "tokens": u["prompt_tokens"] + u["completion_tokens"], "tempo_min": r["elapsed_s"] / 60})
    if not rows:
        sys.exit("Nenhum projeto de teste concluído.")
    df = pd.DataFrame(rows)
    df_f = pd.DataFrame(falhas, columns=["versao", "seed", "t", "fase", "categoria", "repetida"])

    # notas humanas (avaliação cega)
    base = os.path.join(args.out, "avaliacao_cega")
    chave_path = os.path.join(base, ".chave.json")
    tem_humano = False
    if os.path.exists(chave_path):
        chave = json.load(open(chave_path, encoding="utf-8"))
        aval = []
        for fn in sorted(os.listdir(base)):
            if fn.startswith("avaliacao_") and fn.endswith(".csv"):
                for r in read_csv(os.path.join(base, fn)):
                    it = chave.get(r.get("id", "").strip())
                    if not it:
                        continue
                    vals = {c: pd.to_numeric(r.get(c), errors="coerce") for c in CRITERIOS}
                    aval.append({"versao": it["versao"], "seed": it["seed"], "t": it["t"], "avaliador": fn[10:-4], **vals})
        if aval:
            da = pd.DataFrame(aval)
            da["nota_humana"] = da[CRITERIOS].mean(axis=1)
            print(f"Avaliações humanas: {da['avaliador'].nunique()} avaliador(es), {len(da)} notas")
            da.to_csv(os.path.join(out, "avaliacoes_humanas.csv"), index=False)
            media = da.groupby(["versao", "seed", "t"])[CRITERIOS + ["nota_humana"]].mean().reset_index()
            df = df.merge(media, on=["versao", "seed", "t"], how="left")
            tem_humano = True

    df.to_csv(os.path.join(out, "projetos_teste.csv"), index=False)
    def tabela(colunas):  # "média ± desvio", uma linha por versão
        g = df.groupby("versao")[colunas]
        m, d = g.mean(), g.std().fillna(0)
        return m.round(2).astype(str) + " ± " + d.round(2).astype(str)

    qual = (CRITERIOS + ["nota_humana"] if tem_humano else []) + ["nota_llm", "mle_score"]
    proc = ["sucesso_etapas", "falhas", "falhas_repetidas", "tokens", "tempo_min"]
    resumo_q, resumo_p = tabela(qual), tabela(proc)
    por_etapa = (df_f.groupby(["fase", "versao"]).size().unstack(fill_value=0)
                 .reindex(columns=list(VERSOES), fill_value=0)) if len(df_f) else pd.DataFrame()
    n_proj = df.groupby("versao").size()
    if len(por_etapa):
        por_etapa = (por_etapa / n_proj.reindex(por_etapa.columns)).round(2)  # falhas por projeto
    categorias = (df_f.groupby(["categoria", "versao"]).size().unstack(fill_value=0)
                  .reindex(columns=list(VERSOES), fill_value=0)) if len(df_f) else pd.DataFrame()

    md = ["# Resultados (metade de teste)\n", f"Projetos por versão: {n_proj.to_dict()}\n",
          "## Qualidade (média ± desvio)\n", to_md(resumo_q),
          "\n## Falhas, custo e tempo por projeto (média ± desvio)\n", to_md(resumo_p), "\n## Falhas por projeto, por etapa\n",
          to_md(por_etapa) if len(por_etapa) else "(nenhuma falha)",
          "\n## Falhas por categoria (total)\n", to_md(categorias) if len(categorias) else "(nenhuma)"]

    # estatística: comparações pareadas (mesma seed e mesmo tema)
    alvo = "nota_humana" if tem_humano else "nota_llm"
    try:
        from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
        piv = df.pivot_table(index=["seed", "t"], columns="versao", values=alvo).dropna()
        md.append(f"\n## Testes estatísticos ({alvo}, {len(piv)} pares)\n")
        if len(piv) >= 3 and set(VERSOES) <= set(piv.columns):
            st, p = friedmanchisquare(piv["A"], piv["B"], piv["C"])
            md.append(f"- Friedman A×B×C: estatística={st:.2f}, p={p:.3f}")
        for x, y in (("B", "A"), ("C", "A"), ("C", "B")):
            if {x, y} <= set(piv.columns) and len(piv) >= 5 and (piv[x] != piv[y]).any():
                st, p = wilcoxon(piv[x], piv[y])
                md.append(f"- Wilcoxon {x} vs {y}: diferença média={(piv[x] - piv[y]).mean():+.2f}, p={p:.3f}")
        if tem_humano:
            ok = df[["nota_llm", "nota_humana"]].dropna()
            if len(ok) >= 5:
                rho, p = spearmanr(ok["nota_llm"], ok["nota_humana"])
                md.append(f"- Revisor LLM vs. humanos (Spearman): rho={rho:.2f}, p={p:.3f}; "
                          f"média LLM={ok['nota_llm'].mean():.2f}/10 vs. humana={ok['nota_humana'].mean():.2f}/5")
        md.append("\nCom 5 temas por seed o poder estatístico é baixo: trate os p-valores como indicativos.")
    except ImportError:
        md.append("\n(scipy não instalado: testes estatísticos não executados)")

    with open(os.path.join(out, "resumo.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    grafico(df, df_f, tem_humano, os.path.join(out, "grafico_resultados.png"))
    print(f"\nArquivos em {out}")


def grafico(df, df_f, tem_humano, path):
    import matplotlib
    import matplotlib.pyplot as plt
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    cores = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}  # paleta categórica validada, ordem fixa
    vs = [v for v in VERSOES if v in set(df["versao"])]
    paineis = [("nota_humana" if tem_humano else "nota_llm",
                "Nota humana (1–5)" if tem_humano else "Nota do revisor LLM (1–10)"),
               ("falhas_repetidas", "Falhas repetidas do treino, por projeto"),
               ("tokens", "Tokens por projeto")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, titulo) in zip(axes, paineis):
        g = df.groupby("versao")[col]
        m, e = g.mean().reindex(vs).fillna(0), g.std().reindex(vs).fillna(0)
        x = list(range(len(vs)))
        baixo = [min(e[v], m[v]) for v in vs]  # métricas não negativas: a barra de erro não passa de zero
        ax.bar(x, m, yerr=[baixo, list(e)], color=[cores[v] for v in vs], width=0.6, capsize=4,
               edgecolor="white", linewidth=2, error_kw={"elinewidth": 1, "ecolor": "#555"})
        if col == "nota_humana":
            ax.set_ylim(0, 5)
        else:
            ax.set_ylim(bottom=0, top=max(1, float((m + e).max()), float(df[col].max())) * 1.15)
        topo = ax.get_ylim()[1]
        for xi, v in zip(x, vs):  # pontos individuais: com poucos projetos, a média sozinha engana
            pts = df.loc[df["versao"] == v, col].dropna()
            ax.scatter([xi] * len(pts), pts, color="#333", s=14, zorder=3, alpha=0.6)
            txt = f"{m[v]:.2f}" if col != "tokens" else f"{m[v]/1000:.0f}k"
            dentro = m[v] > 0.12 * topo
            ax.annotate(txt, (xi, 0 if dentro else m[v]), xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=9, color="white" if dentro else "#333", fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{v}\n{VERSOES[v]}" for v in vs], fontsize=9)
        ax.set_title(titulo, loc="left", fontsize=11)
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etapa", choices=["treino", "memoria", "teste", "avaliacao-cega", "analise"])
    ap.add_argument("--fonte", choices=["llm", "humano"], help="etapa memoria: origem das lições")
    ap.add_argument("--versao", choices=list(VERSOES), help="etapa teste: A, B ou C")
    ap.add_argument("--topics", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "topics.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--num_papers", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--mle_steps", type=int, default=3)
    ap.add_argument("--mock", action="store_true", help="LLM, feedback e notas simulados (teste do fluxo)")
    ap.add_argument("--refazer", action="store_true", help="avaliacao-cega: gerar de novo")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.etapa == "memoria" and not args.fonte:
        ap.error("informe --fonte llm ou --fonte humano")
    if args.etapa == "teste" and not args.versao:
        ap.error("informe --versao A, B ou C")
    os.makedirs(args.out, exist_ok=True)
    # o código gerado pelo agente usa o mesmo modelo quando o experimento precisa de um LLM
    os.environ["AGENTLAB_LLM_MODEL"] = args.model
    if args.base_url:
        os.environ["AGENTLAB_LLM_BASE_URL"] = args.base_url
    {"treino": cmd_treino, "memoria": cmd_memoria, "teste": cmd_teste,
     "avaliacao-cega": cmd_avaliacao, "analise": cmd_analise}[args.etapa](args)


if __name__ == "__main__":
    main()
