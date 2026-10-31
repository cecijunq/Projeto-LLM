"""
Sincroniza o código entre o notebook do Colab e a pasta agent_lab/ (para rodar localmente).

  python sync_notebook.py extrair   # notebook -> agent_lab/  (cada célula %%writefile vira um arquivo)
  python sync_notebook.py gravar    # agent_lab/ -> notebook  (antes de subir o notebook para o Colab)
  python sync_notebook.py checar    # mostra quais arquivos diferem entre os dois

Na pasta local, os arquivos de agent_lab/ são a fonte da verdade: edite-os e rode `gravar` antes de
subir o notebook. `extrair` recusa sobrescrever um arquivo local diferente do notebook (use --forcar).
"""
from __future__ import annotations

import json
import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(RAIZ, "Projeto_AgentLab_Memoria_Colab.ipynb")
PASTA = os.path.join(RAIZ, "agent_lab")
PREFIXO = "/content/agent_lab/"


def celulas(nb: dict):
    """(célula, caminho local, conteúdo) para cada célula %%writefile do código."""
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and src.startswith("%%writefile " + PREFIXO):
            primeira, _, corpo = src.partition("\n")
            rel = primeira.split()[1][len(PREFIXO):]
            yield c, os.path.join(PASTA, rel), corpo


def ler(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("extrair", "gravar", "checar"):
        sys.exit(__doc__)
    acao, forcar = sys.argv[1], "--forcar" in sys.argv
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    if acao == "checar":
        for _, path, corpo in celulas(nb):
            local = ler(path)
            estado = "ausente" if local is None else ("igual" if local == corpo else "DIFERENTE")
            print(f"{estado:10s} {os.path.relpath(path, RAIZ)}")
        return

    if acao == "extrair":
        conflitos = [p for _, p, corpo in celulas(nb) if ler(p) not in (None, corpo)]
        if conflitos and not forcar:
            sys.exit("Arquivos locais diferentes do notebook (rode 'gravar' ou use --forcar para descartá-los):\n  "
                     + "\n  ".join(os.path.relpath(p, RAIZ) for p in conflitos))
        for _, path, corpo in celulas(nb):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(corpo)
            print("extraído:", os.path.relpath(path, RAIZ))
        return

    n = 0
    for c, path, corpo in celulas(nb):
        local = ler(path)
        if local is None or local == corpo:
            continue
        primeira = "".join(c["source"]).partition("\n")[0]
        c["source"] = (primeira + "\n" + local).splitlines(keepends=True)
        n += 1
        print("gravado no notebook:", os.path.relpath(path, RAIZ))
    if n:
        with open(NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(f"{n} célula(s) atualizada(s).")


if __name__ == "__main__":
    main()
