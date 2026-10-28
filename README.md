# Agent Laboratory + memória entre projetos

Implementação experimental das versões A/B/C descritas na proposta do projeto:

- A: Agent Laboratory sem memória de longo prazo.
- B: memória construída a partir do feedback do revisor LLM.
- C: memória construída a partir do feedback humano.

O código executável está em `agent_lab_memory/` e `projeto.py`. O notebook em
`notebooks/` preserva o fluxo original de Google Colab; para executar no macOS,
use os comandos abaixo, sem rodar as células que instalam dependências do Colab.

## Requisitos para macOS Apple Silicon

- Python 3.11 ou 3.12 recomendado.
- Ollama para macOS, com o modelo `qwen2.5:7b`.
- Para o MacBook M3 Pro com 18 GB de memória unificada, comece com 7B e contexto
  de 8k. Evite executar outros modelos locais simultaneamente. O modelo 14B e
  contextos maiores podem pressionar a memória e reduzir o desempenho.
- O Ollama usa Metal automaticamente no Apple Silicon. Não é necessário CUDA,
  `nvidia-smi`, `apt-get` nem Google Drive.

## Instalação

Instale Ollama pelo aplicativo oficial e baixe o modelo:

```bash
ollama pull qwen2.5:7b
ollama list
```

Crie o ambiente Python e instale as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O aplicativo Ollama normalmente mantém o servidor local ativo em
`http://localhost:11434`. Se não estiver ativo, abra o aplicativo Ollama antes
de executar o projeto. Para limitar o contexto a 8k, configure o servidor com
`OLLAMA_CONTEXT_LENGTH=8192` antes de iniciá-lo; não inicie uma segunda instância
se o aplicativo já estiver atendendo nessa porta.

Configure a API compatível com OpenAI do Ollama. A chave é apenas um valor local
exigido pelo cliente e não deve ser uma credencial real:

```bash
export OPENAI_API_KEY=ollama
```

## Verificação sem chamadas ao modelo

Os testes unitários não acessam serviços externos:

```bash
python -m unittest discover -s tests -v
```

O modo mock exercita o pipeline e grava artefatos temporários em `/tmp`:

```bash
python projeto.py treino --mock --out /tmp/agentlab-mock --max_steps 6 --num_papers 2
python projeto.py memoria --fonte llm --mock --out /tmp/agentlab-mock --max_steps 6 --num_papers 2
python projeto.py teste --versao A --mock --out /tmp/agentlab-mock --max_steps 6 --num_papers 2
python projeto.py teste --versao B --mock --out /tmp/agentlab-mock --max_steps 6 --num_papers 2
```

O mock verifica encadeamento e persistência, mas **não** mede qualidade do
modelo, custo, velocidade de inferência ou benefício real da memória.

## Execução local real

Comece com uma configuração curta e um diretório de saída separado:

```bash
python projeto.py treino \
  --model qwen2.5:7b --provider openai \
  --base_url http://localhost:11434/v1 \
  --out results/piloto --seeds 1 --num_papers 2 --max_steps 6 --mle_steps 1
```

Após confirmar uma execução completa, use as mesmas opções em cada etapa. Para
continuar o protocolo A/B/C, execute na ordem `treino`, `memoria --fonte llm`,
`teste --versao A`, `teste --versao B`, preencha
`results/piloto/feedback_humano.csv`, depois `memoria --fonte humano`,
`teste --versao C`, `avaliacao-cega` e `analise`. O default do CLI grava em
`results/`; `--out` permite escolher outro local.

Um projeto completo faz muitas chamadas sequenciais ao modelo. Meça primeiro o
tempo de um tema em modo curto; não estime a duração total apenas pelo tempo de
uma resposta isolada.

## Artefatos e segurança

Cada execução salva relatório, plano, códigos, resultados, métricas e logs no
diretório indicado por `--out`. Esses arquivos podem conter prompts e respostas
do modelo. Não os publique sem revisão.

O agente executa código Python produzido por LLM em subprocessos. Isso não é um
sandbox: rode experimentos em uma conta/ambiente sem credenciais, dados ou
arquivos sensíveis acessíveis. Não use `--mock` como evidência de resultados
científicos.

Antes da entrega acadêmica, registre os resultados verificados, as limitações,
as contribuições de cada integrante e a declaração de uso de IA exigida pela
disciplina. O grupo deve conseguir explicar e validar todo o código submetido.