# Agent Laboratory + memória entre projetos

Projeto final da disciplina **Large Language Models (UFMG, 2026/2)**, Grupo 1: Arthur Siríaco, Cecília Junqueira e Déborah Yamamoto.

## Resumo

O **Agent Laboratory** (Schmidgall et al., 2025, [arXiv:2501.04227](https://arxiv.org/abs/2501.04227)) recebe uma ideia de pesquisa e, com vários agentes LLM, faz a revisão de literatura, planeja e roda experimentos e escreve um relatório. No fim de cada projeto, tudo o que ele aprendeu é descartado. O próprio artigo mostra três problemas que motivam este trabalho: o revisor automático é otimista demais (6,1/10 contra 3,8 dos humanos), o feedback humano melhora os resultados (+0,58 na nota geral no modo co-pilot) e os mesmos erros se repetem de um projeto para outro.

Este projeto acrescenta ao agente uma **memória de lições que sobrevive entre projetos**. A partir do feedback recebido por um projeto, um LLM extrai lições no formato "problema → como evitar". Os projetos seguintes recuperam as lições mais relevantes para cada fase e as recebem no prompt.

- **QP1.** Guardar lições de projetos anteriores faz o agente se sair melhor nos próximos?
- **QP2.** Importa de onde vem a lição: do revisor automático (LLM) ou de um humano?
- **Hipótese:** C > B > A.

| Versão | Memória |
|---|---|
| **A** | sem memória (baseline: nossa reimplementação simplificada do Agent Laboratory, **não** o código original dos autores) |
| **B** | lições extraídas do feedback do **revisor LLM** |
| **C** | lições extraídas do **feedback humano** (do grupo) |

### Protocolo

1. **Treino.** Os 5 temas de treino rodam uma vez, sem memória.
2. **Memórias.** Os mesmos relatórios de treino recebem dois tipos de feedback. A revisão do LLM gera a memória B, e o feedback humano gera a memória C. As duas passam pelo mesmo extrator de lições, com a mesma seed, então só o texto do feedback muda. É isso que isola a QP2.
3. **Teste.** Os 5 temas de teste rodam nas versões A, B e C. A memória fica **congelada** (somente leitura), e a mesma seed de geração é usada para cada par (seed, tema).
4. **Avaliação cega.** Os relatórios de teste são anonimizados e recebem notas de 1 a 5 nos critérios do artigo (`qualidade_experimental`, `qualidade_relatorio`, `utilidade`).
5. **Análise.** Compara a nota humana e a nota do revisor LLM, as falhas por etapa (incluindo as **repetidas** do treino), o custo em tokens e o tempo. Os testes estatísticos são Friedman (A×B×C), Wilcoxon pareado e a correlação de Spearman entre a nota do LLM e a humana.

Os 10 temas estão em `topics.json`: 5 vêm da Seção 4.1 do artigo e 5 são novos.

## Estrutura do código

A fonte do código é o notebook **`Projeto_AgentLab_Memoria_Colab.ipynb`**. As células da seção 0.6 gravam os arquivos abaixo em `/content/agent_lab` com `%%writefile`:

```
agent_lab_memory/
  __init__.py
  llm.py          # cliente LLM (OpenAI/Ollama/Anthropic) e MockLLM; contagem de tokens; extract_command() tolerante a formato
  memory.py       # LongTermMemory: retrieve = 0,6·similaridade + 0,25·importância + 0,15·recência + 0,1 se for da mesma fase
                  # extract_lessons / store_lessons / format_llm_review / project_facts
  failures.py     # detecção automática de falhas por (fase, categoria)
  tools.py        # arXiv, busca e verificação de datasets do HuggingFace, execução de código em subprocesso com limites
  correcoes.py    # versão D (exploratória, desligada em A/B/C)
  agents.py       # agentes PhD / Postdoc / ML Engineer (prompts condensados do Apêndice B do artigo)
  solvers.py      # MLESolver (gera, repara e pontua código), PaperSolver (relatório) e revisor automático
  laboratory.py   # orquestrador das fases; a memória entra nos pontos marcados com # [MEMÓRIA]
projeto.py        # CLI do protocolo: treino | memoria | teste | avaliacao-cega | analise
topics.json       # 5 temas de treino + 5 de teste
```

**Fases do agente:** `literature_review → plan_formulation → data_preparation → running_experiments → results_interpretation → report_writing → report_refinement`. Se `data_preparation` ou `running_experiments` falhar, o projeto é encerrado. Na fase de refinamento, o agente pode voltar a uma fase anterior no máximo uma vez.

## Configuração dos resultados principais

Estes são os valores da célula 0.1 do notebook. Use **os mesmos em todas as etapas**: o `projeto.py` grava a configuração em cada `result.json` e para com erro se encontrar resultados de outra configuração.

| Parâmetro | Valor |
|---|---|
| Modelo | `qwen3-nothink:8b` (o `qwen3:8b` sem raciocínio, criado na preparação) |
| Servidor | Ollama, API compatível com OpenAI em `http://127.0.0.1:11434/v1` |
| Contexto do Ollama | 16384 tokens |
| `--seeds` | 3 (repetições do protocolo inteiro) |
| `--num_papers` / `--max_steps` / `--mle_steps` | 5 / 20 / 3 (`MODO_RAPIDO = False`) |
| Hardware | Google Colab com GPU T4 |
| Seeds de geração | treino: `100·s + i`; teste: `500000 + 100·s + i` (iguais em A, B e C); extrator de lições: `777000 + s` |

Com 3 seeds e 5 temas, são 15 projetos de treino e 15 de teste por versão, ou seja, 45 relatórios na avaliação cega.

## Instalação

### Opção 1: Google Colab (recomendada; é o ambiente do notebook)

1. Abra `Projeto_AgentLab_Memoria_Colab.ipynb` no Colab.
2. Escolha `Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4`.
3. Rode a **Parte 0 inteira**, que faz o seguinte:
   - define a configuração (0.1) e monta o Google Drive (0.2). Os resultados ficam em `/content/drive/MyDrive/agent_lab_projeto`;
   - confere a GPU (0.3);
   - instala `arxiv pypdf datasets huggingface_hub` e o Ollama (0.4);
   - sobe o servidor Ollama e cria o modelo `qwen3-nothink:8b` (0.5);
   - grava o código do projeto em `/content/agent_lab` (0.6);
   - roda os testes rápidos (0.7): fluxo com LLM simulado, formato de resposta do modelo real, arXiv e HuggingFace.

**Em toda sessão nova do Colab**, rode a Parte 0 de novo e continue a partir da etapa em que parou. As etapas são retomáveis: projetos que já têm `result.json` são pulados.

### Opção 2: máquina local (Linux com GPU NVIDIA)

Esta opção reproduz o que a Parte 0 faz. Requer Python 3.10 ou superior e o [Ollama](https://ollama.com).

**1. Extraia o código do notebook.** O script abaixo grava os arquivos das células `%%writefile` no diretório atual:

```bash
python - Projeto_AgentLab_Memoria_Colab.ipynb <<'EOF'
import json, os, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c["source"])
    if c["cell_type"] == "code" and src.startswith("%%writefile"):
        first, _, body = src.partition("\n")
        path = first.split(maxsplit=1)[1].strip().replace("/content/agent_lab/", "")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "w", encoding="utf-8").write(body)
        print("gravado:", path)
EOF
```

**2. Instale as dependências Python.** No Colab, `openai`, `sentence-transformers`, `pandas`, `scipy` e `matplotlib` já vêm instalados. Localmente, instale todos:

```bash
python -m venv .venv && source .venv/bin/activate
pip install openai arxiv pypdf datasets huggingface_hub sentence-transformers pandas scipy matplotlib tabulate
```

O `sentence-transformers` é importante. Sem ele, a memória troca os embeddings `all-MiniLM-L6-v2` por TF-IDF sem avisar, o que muda a recuperação e a deduplicação das lições. Para conferir:

```bash
python -c "from agent_lab_memory import Embedder; print('denso' if Embedder().dense is not None else 'TF-IDF')"
```

**3. Suba o Ollama e crie o modelo sem raciocínio.** O passo abaixo é o mesmo da célula 0.5 do notebook: troca uma condição no template do `qwen3:8b` para que o bloco `<think>` venha sempre vazio.

```bash
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_KEEP_ALIVE=-1 OLLAMA_HOST=127.0.0.1:11434 ollama serve > ollama.log 2>&1 &

ollama pull qwen3:8b
ollama show qwen3:8b --modelfile > Modelfile.q3
python - <<'EOF'
alvo = "{{ if and $.IsThinkSet (not $.Think) -}}"
s = open("Modelfile.q3").read()
assert alvo in s, "o template do qwen3 mudou: confira a criação do qwen3-nothink"
open("Modelfile.q3", "w").write(s.replace(alvo, "{{ if true -}}"))
EOF
ollama create qwen3-nothink:8b -f Modelfile.q3

export OPENAI_API_KEY=ollama   # valor local exigido pelo cliente, não é uma credencial
```

As variáveis `AGENTLAB_LLM_MODEL` e `AGENTLAB_LLM_BASE_URL`, que o código gerado pelo agente usa para consultar o modelo, são definidas pelo `projeto.py` a partir de `--model` e `--base_url`.

## Validação rápida (sem LLM)

O modo `--mock` testa o encadeamento das etapas e a persistência em segundos, com respostas simuladas. Estes são os mesmos comandos da célula 0.7:

```bash
for e in "treino" "memoria --fonte llm" "teste --versao A" "teste --versao B"; do
  python projeto.py $e --mock --out /tmp/teste_mock --max_steps 6 --num_papers 2 > /dev/null && echo "ok: $e"
done
```

As outras etapas também aceitam `--mock`: o feedback humano e as notas dos avaliadores são simulados. O mock **não** mede a qualidade do modelo, o custo nem o benefício real da memória, e não deve ser usado como resultado.

## Reproduzindo os resultados principais

Defina a configuração uma vez. No Colab, a célula 0.1 já monta `ARGS`, e as células usam `{ARGS}`.

```bash
OUT=resultados_projeto   # no Colab: /content/drive/MyDrive/agent_lab_projeto
ARGS="--model qwen3-nothink:8b --base_url http://127.0.0.1:11434/v1 --seeds 3 --num_papers 5 --max_steps 20 --mle_steps 3 --out $OUT"
```

Rode as etapas nesta ordem. Na tabela, "célula" é a do notebook.

| # | Célula | Comando | O que gera |
|---|---|---|---|
| 1 | 1.1 | `python projeto.py treino $ARGS --verbose` | `treino/seed*/t*/` e `feedback_humano.csv` vazio |
| 2 | 1.2 | `python projeto.py memoria --fonte llm $ARGS` | `memoria/B_seed*.json` e `memoria/licoes_B_seed*.md` |
| 3 | 1.3 | `python projeto.py teste --versao A $ARGS --verbose` | `teste/A/seed*/t*/` |
| 4 | 1.4 | `python projeto.py teste --versao B $ARGS --verbose` | `teste/B/seed*/t*/` |
| 5 | 2.1 | *(manual)* preencher `feedback_humano.csv` | feedback do grupo sobre cada projeto de treino |
| 6 | 2.2 | `python projeto.py memoria --fonte humano $ARGS` | `memoria/C_seed*.json` e `memoria/licoes_C_seed*.md` |
| 7 | 2.3 | `python projeto.py teste --versao C $ARGS --verbose` | `teste/C/seed*/t*/` |
| 8 | 3.1 | `python projeto.py avaliacao-cega $ARGS` | `avaliacao_cega/relatorios/R01.md…`, `planilha_modelo.csv` e `.chave.json` |
| 9 | 3.2 | *(manual)* cada avaliador preenche `avaliacao_<nome>.csv` | as notas humanas |
| 10 | 3.3 | `python projeto.py analise $ARGS` | `analise/resumo.md`, as tabelas CSV e `grafico_resultados.png` |

A célula 1.5 roda `analise` logo depois do passo 4, para ter um resultado parcial A vs. B só com métricas automáticas. Esse resultado usa a nota do revisor LLM, que o artigo mostra ser otimista.

### Etapas manuais

**Feedback humano (passo 5).** Para cada projeto de treino, o grupo lê o arquivo indicado na coluna `arquivo_para_ler` (`treino/seed*/t*/resumo_para_feedback.md`) e preenche `nota_1a5`, `o_que_deu_errado` e `como_evitar`. Pode ser em português, porque o extrator traduz as lições. Há duas formas de preencher:

- na célula 2.1 do notebook, que mostra cada projeto pendente e salva a cada resposta;
- editando o CSV no Excel ou no Google Sheets. O separador pode ser `;` ou `,`.

O passo 6 aborta se algum projeto ficar sem feedback.

**Avaliação cega (passo 9).** Cada integrante avalia separadamente. Use a célula 3.2, ou copie `avaliacao_cega/planilha_modelo.csv` para `avaliacao_cega/avaliacao_<nome>.csv` e preencha `qualidade_experimental`, `qualidade_relatorio` e `utilidade` (1–5). **Não abra `.chave.json` antes de todas as notas estarem prontas**: ela revela a versão de cada relatório.

### Onde estão os resultados

| Caminho (em `$OUT`) | Conteúdo |
|---|---|
| `treino/seed*/t*/` | projetos de treino: `resumo_para_feedback.md`, `report.md`, códigos, `result.json` e `llm_calls.jsonl` |
| `feedback_humano.csv` | o feedback do grupo (entrada da memória C) |
| `memoria/` | as memórias `B_seed*.json` e `C_seed*.json`, com as lições legíveis em `licoes_*.md` |
| `teste/{A,B,C}/seed*/t*/` | projetos de teste de cada versão |
| `avaliacao_cega/` | relatórios anonimizados, planilhas dos avaliadores e a chave |
| `analise/resumo.md` | **resultado principal**: qualidade (média ± desvio), falhas por etapa e categoria, tokens, tempo e testes estatísticos |
| `analise/grafico_resultados.png` | nota humana, falhas repetidas e tokens por versão |
| `analise/projetos_teste.csv`, `analise/avaliacoes_humanas.csv` | os dados de cada projeto e de cada avaliação |

A métrica principal é a `nota_humana`, que é a média dos três critérios entre os avaliadores. Sem avaliações humanas, a análise usa a nota do revisor LLM.

## Reprodutibilidade e invariantes

- **Mesma configuração em tudo.** Para mudar o modelo, os passos ou as seeds, é preciso apagar os resultados antigos (por exemplo, `rm -rf $OUT/teste`) e refazer as etapas afetadas.
- **Retomada.** Se a execução cair, basta rodar o mesmo comando de novo: os projetos concluídos são pulados. `memoria` e `avaliacao-cega` também não sobrescrevem o que já existe; para regerar, use `--refazer`. Na avaliação cega, isso embaralha os relatórios outra vez.
- **Falhas são dado do experimento.** Não rode de novo um projeto de teste só porque ele falhou. As falhas entram na métrica de falhas repetidas.
- **Variação residual.** Mesmo com seed fixa, a geração no Ollama e as respostas do arXiv e do HuggingFace podem variar entre execuções. O código gerado pelo agente também roda de verdade, então os números exatos não são garantidamente idênticos.
- **Poder estatístico baixo.** Com 5 temas por seed, os p-valores são indicativos. Vale olhar também as lições em `memoria/licoes_*.md` e as falhas repetidas.

### Versão D (exploratória, fora da comparação A/B/C)

`python projeto.py teste --versao D $ARGS` roda a versão A com as correções de `correcoes.py`, que são instruções escritas a partir dos erros vistos no treino. Serve como teto de referência para o que correções manuais conseguiriam. Os resultados ficam em `teste/D/` e não entram na `analise`.

## Diagnóstico

Se uma fase aparecer como `FALHOU` no log, a célula "Se aparecer `FALHOU` em alguma fase" (logo depois da 1.5) mostra as últimas respostas brutas do modelo no projeto mais recente. O mesmo conteúdo está em `llm_calls.jsonl`. Em geral:

- **muitos comandos inválidos** indicam que o modelo não segue o formato de comando;
- **0 comandos inválidos, mas falhou** indica que faltaram passos.

Lembre que trocar o modelo ou os passos exige refazer os resultados (ver Invariantes).

## Segurança e limitações

- O agente executa código Python gerado pelo LLM em subprocessos, com limites de tempo (300 s na preparação de dados e 600 s nos experimentos), RAM (6 GB) e saída (50 MB). **Isso não é um sandbox.** Rode em um ambiente sem credenciais nem dados sensíveis, como o Colab ou uma VM descartável.
- Os artefatos (`llm_calls.jsonl`, relatórios e códigos) contêm prompts e respostas do modelo. Revise antes de publicar.
- A versão A é uma **reimplementação simplificada** do Agent Laboratory, e não o código original dos autores.
- O relatório final deve registrar os resultados verificados, as limitações, a contribuição de cada integrante e a declaração de uso de IA exigida pela disciplina.
