"""Acompanha uma etapa em andamento (treino ou teste) só lendo a pasta de resultados; não interfere na execução.

  python progresso.py ../resultados_oficial/teste/C            # mostra uma vez
  python progresso.py ../resultados_oficial/teste/C --watch 60 # atualiza a cada 60 s (Ctrl+C para sair)

Projeto concluído = tem result.json. A fase do projeto em andamento é deduzida dos arquivos que ele já gravou.
A estimativa usa o tempo médio (elapsed_s) dos projetos já concluídos nessa mesma pasta.
"""
import argparse
import json
import os
import time

# arquivo gravado ao terminar cada fase, na ordem do laboratório
FASES = [("literature.txt", "revisão da literatura"), ("plan.txt", "plano"),
         ("dataset_code.py", "preparação dos dados"), ("results.txt", "experimentos"),
         ("interpretation.txt", "interpretação"), ("report.md", "relatório")]


def fmt(seg: float) -> str:
    h, m = divmod(int(seg) // 60, 60)
    return f"{h}h{m:02d}" if h else f"{m} min"


def barra(feito: float, total: int, largura: int = 30) -> str:
    n = round(largura * feito / total) if total else 0
    return "█" * n + "░" * (largura - n)


def fase_atual(d: str) -> str:
    """Os arquivos das fases só são gravados no fim do projeto: a fase vem da última chamada ao LLM."""
    try:
        with open(os.path.join(d, "llm_calls.jsonl"), "rb") as f:
            f.seek(max(0, os.path.getsize(f.name) - 200_000))
            sistema = json.loads(f.read().decode("utf-8", "replace").strip().splitlines()[-1]).get("system", "")
    except (OSError, ValueError, IndexError):
        return "?"
    t = sistema[sistema.find("Task instructions:"):][:300].lower()
    for chave, n in (("literature review", 1), ("design a simple", 2), ("data preparation", 3),
                     ("ml engineer", 3), ("interpret", 5), ("report", 6), ("latex", 6)):
        if chave in t:
            return f"{FASES[n - 1][1]} ({n}/{len(FASES)})"
    return f"{FASES[3][1]} (4/{len(FASES)})"


def status(base: str, seeds: int, temas: int) -> str:
    total = seeds * temas
    tempos, em_andamento = [], []
    for s in range(seeds):
        for i in range(temas):
            d = os.path.join(base, f"seed{s}", f"t{i}")
            p = os.path.join(d, "result.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    tempos.append(json.load(f).get("elapsed_s", 0))
            elif os.path.isdir(d) and os.listdir(d):
                mts = [os.path.getmtime(os.path.join(d, x)) for x in os.listdir(d)]
                em_andamento.append((f"seed{s}/t{i}", fase_atual(d), time.time() - min(mts), time.time() - max(mts)))
    feitos = len(tempos)
    linhas = [f"{base}", f"[{barra(feitos, total)}] {feitos}/{total} projetos ({100 * feitos / total:.0f}%)"]
    for pid, fase, desde, parado in em_andamento:
        if parado > 30 * 60:  # uma chamada ao LLM não leva 30 min: a execução caiu
            linhas.append(f"  ⚠ {pid} parou na fase {fase}: nada gravado há {fmt(parado)}. "
                          f"Rode a etapa de novo (ela retoma; este projeto recomeça do zero).")
        else:
            linhas.append(f"  em andamento: {pid} · fase: {fase} · começou há {fmt(desde)} · "
                          f"última atividade há {fmt(parado)}")
    if tempos:
        media = sum(tempos) / len(tempos)
        linhas.append(f"  tempo médio por projeto: {fmt(media)} · faltam ~{fmt(media * (total - feitos))}"
                      if feitos < total else "  concluído!")
    return "\n".join(linhas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta", help="ex.: ../resultados_oficial/teste/C")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temas", type=int, default=5)
    ap.add_argument("--watch", type=int, default=0, help="atualiza a cada N segundos")
    a = ap.parse_args()
    while True:
        txt = status(a.pasta, a.seeds, a.temas)
        if a.watch:
            print("\033[2J\033[H" + time.strftime("%H:%M:%S ") + txt, flush=True)
            time.sleep(a.watch)
        else:
            print(txt)
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
