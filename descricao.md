# Agent Laboratory + memória entre projetos

Projeto final da disciplina **Large Language Models (UFMG, 2026/2)**, Grupo 1: Arthur Siríaco, Cecília Junqueira e Déborah Yamamoto.

Este arquivo dá contexto para assistentes de código (Claude no VS Code). Leia antes de propor mudanças.

## Objetivo

O **Agent Laboratory** (Schmidgall et al., 2025, arXiv:2501.04227) recebe uma ideia de pesquisa e, com vários agentes LLM, faz a revisão de literatura, roda os experimentos e escreve o relatório. Ao fim de cada projeto, **tudo o que ele aprendeu é descartado**.

O artigo relata três coisas que motivam este trabalho:

- O revisor automático é otimista demais: dá nota 6,1/10, enquanto revisores humanos dão 3,8.
- O feedback humano a cada etapa (modo co-pilot) melhora a nota geral em +0,58.
- Os mesmos erros se repetem entre projetos, como buscas em loop e experimentos com 0% de acurácia.

Por isso, adicionamos uma **memória de lições que sobrevive entre projetos**. Ao fim de cada projeto, guardamos "o que deu errado e como evitar". Os projetos seguintes consultam essas lições.

### Perguntas de pesquisa

- **QP1.** Guardar lições de projetos anteriores faz o agente se sair melhor nos próximos?
- **QP2.** Importa de onde vem a lição: do revisor automático (LLM) ou de um humano?

**Hipótese:** C > B > A.

## Desenho experimental

| Versão | Memória |
|---|---|
| **A** | sem memória (baseline: nossa reimplementação do Agent Laboratory) |
| **B** | lições extraídas do feedback do **revisor LLM** |
| **C** | lições extraídas do **feedback humano** (do grupo) |

**Protocolo** (temas em `topics.json`):

1. **Treino.** Os 5 temas de treino rodam **uma vez**, sem memória.
2. **Construção das memórias.** Os **mesmos** relatórios de treino recebem os dois tipos de feedback:
   - a revisão do LLM gera a memória B;
   - o `feedback_humano.csv` gera a memória C.
3. **Teste.** Os 5 temas de teste rodam nas versões A, B e C, com a memória **congelada** (somente leitura).
4. **Avaliação.** A avaliação humana é **cega**: os relatórios são anonimizados e recebem notas de 1 a 5 nos critérios do artigo (`qualidade_experimental`, `qualidade_relatorio`, `utilidade`).
5. **Análise.**
   - Qualidade: nota humana e nota do revisor LLM.
   - Falhas por etapa, incluindo as **falhas repetidas** do treino.
   - Custo (tokens) e tempo por projeto.
   - Testes estatísticos: Friedman, Wilcoxon pareado e correlação de Spearman entre a nota do LLM e a humana.

### Invariantes que NÃO podem ser quebrados

Estas regras garantem que a comparação seja justa:

- **Mesmo modelo, mesmos passos e mesma configuração** em A, B e C. Mudar qualquer um desses itens exige apagar e refazer os resultados afetados.
- **B e C vêm dos mesmos projetos de treino e do mesmo extrator de lições** (`extract_lessons`, com a mesma seed). A única diferença é o texto do feedback. É isso que isola a QP2.
- **Memória somente leitura no teste.** Os projetos de teste partem do mesmo estado de memória, e a ordem dos temas não interfere.
- **Mesma seed de geração por (seed, tema)** em A, B e C.
- **A avaliação humana é cega.** Não exibir nem usar o conteúdo de `avaliacao_cega/.chave.json` antes das notas estarem completas.
- **Falhas são dado do experimento.** Não "consertar" um projeto de teste rodando de novo só porque ele falhou.

## Ambiente

- Google Colab com **GPU T4 (15 GB)**.
- LLM local via **Ollama**, com API compatível com OpenAI em `http://127.0.0.1:11434/v1`.
- Modelo padrão `qwen2.5:7b`; a alternativa é `qwen2.5:14b`, melhor e mais lenta.
- Os resultados ficam no Google Drive (`/content/drive/MyDrive/agent_lab_projeto`). As etapas são retomáveis: projetos com `result.json` são pulados.
- `--mock` em qualquer etapa testa o fluxo sem LLM.

## Estrutura do código

O notebook `Projeto_AgentLab_Memoria_Colab.ipynb` grava estes arquivos com `%%writefile` em `/content/agent_lab`:

```
agent_lab_memory/
  llm.py         # LLM (OpenAI/Ollama/Anthropic) e MockLLM; contagem de tokens; extract_command() tolerante a formato
  memory.py      # LongTermMemory (retrieve = similaridade + importância + recência + bônus de fase);
                 # extract_lessons / store_lessons / format_llm_review / project_facts
  failures.py    # detecção automática de falhas por (fase, categoria)
  tools.py       # busca no arXiv, busca de datasets no HuggingFace, execução de código em subprocesso
  agents.py      # agentes PhD / Postdoc / ML Engineer (prompts condensados do Apêndice B do artigo)
  solvers.py     # MLESolver (gera, repara e pontua código), PaperSolver (relatório) e revisor automático
  laboratory.py  # orquestrador das fases; a memória entra nos pontos marcados com # [MEMÓRIA]
projeto.py       # CLI do protocolo: treino | memoria | teste | avaliacao-cega | analise
topics.json      # 5 temas de treino + 5 de teste (5 deles vêm da Seção 4.1 do artigo)
```

**Fases do agente:** `literature_review → plan_formulation → data_preparation → running_experiments → results_interpretation → report_writing → (report_refinement)`. Se `data_preparation` ou `running_experiments` falhar, o projeto é encerrado.

**Ao editar o código:** a fonte da verdade são as células `%%writefile` do notebook. Se o código for extraído para arquivos `.py` no repositório, manter os dois sincronizados ou passar a usar só os arquivos (e o notebook apenas executa).

## Status atual (27/10/2026)

- O fluxo completo passa no teste com `--mock`. O modelo real responde no formato e o arXiv funciona.
- **Treino incompleto:**
  - `t0` (vieses cognitivos) falhou em `data_preparation` depois de 10 passos. A última resposta do modelo foi `load_dataset('rotten_tomatoes', ...)`, um dataset irrelevante para o tema.
  - `t1` foi interrompido manualmente durante `plan_formulation`.
- O teste A foi iniciado, mas a memória B ainda não existe.
- O notebook ainda está com `MODO_RAPIDO = True`, que serve só para validar o fluxo.

## Problemas conhecidos e próximos passos

1. **Definir a configuração final** (modelo, `MAX_STEPS`, `MODO_RAPIDO=False`, `SEEDS`) e apagar o que foi rodado em outra configuração.
2. **Diagnosticar a falha do t0.** Ver `errors` em `treino/seed0/t0/result.json` e `llm_calls.jsonl`.
3. **Escolha de datasets.** `suggest_datasets()` empurra os temas de avaliação de LLM para datasets aleatórios do HuggingFace. Considerar permitir que o agente monte um conjunto pequeno de exemplos à mão nesses temas.
4. **Timeout nas chamadas ao LLM.** O cliente OpenAI em `llm.py` não define timeout, e uma chamada travada no Ollama prende a execução. Sugestão: `OpenAI(..., timeout=180)`, já que `query` tenta de novo 4 vezes.
5. **Risco para a QP2.** Projeto de treino que não chega ao relatório não tem revisão LLM. Nesse caso, as lições de B vêm só dos "fatos do projeto", que B e C também recebem. A maior parte do treino precisa chegar ao relatório; um modelo maior ou mais passos podem ajudar.
6. **Poder estatístico baixo** (1 seed × 5 temas). Tratar os p-valores como indicativos e dar peso à análise qualitativa das lições (`memoria/licoes_*.md`) e às falhas repetidas. `SEEDS=2` ajuda, se o tempo de GPU permitir.
7. **No relatório final**, deixar claro que a versão A é uma **reimplementação simplificada**, e não o código original do Agent Laboratory.

## Cronograma

- ~~23/10~~ (atrasado): treino completo, memória B, testes A e B.
- **13/11 (entrega final):** feedback humano e memória C, teste C, avaliação cega, análise e relatório.

## Como ajudar

- O trabalho é de grupo e de aprendizado. Explique o raciocínio das mudanças, não só o código.
- Priorize mudanças pequenas, que não quebrem os invariantes acima nem a retomada das etapas.
- Antes de rodar algo caro (treino ou teste com o LLM real), valide com `--mock`.
- Respostas em português.