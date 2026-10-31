#!/usr/bin/env bash
# Roda uma etapa do protocolo localmente (WSL + GPU), com a mesma configuração da célula 0.1 do notebook.
#   ./rodar_local.sh treino
#   ./rodar_local.sh memoria --fonte llm
#   ./rodar_local.sh teste --versao A
#   ./rodar_local.sh treino --mock          (fluxo sem LLM)
# Variáveis opcionais: MODELO (padrão qwen3-nothink:8b), MODO_RAPIDO (padrão 0), SAIDA (pasta dos resultados).
set -euo pipefail
cd "$(dirname "$0")"
MODELO="${MODELO:-qwen3-nothink:8b}"
MODO_RAPIDO="${MODO_RAPIDO:-0}"
SAIDA="${SAIDA:-$PWD/resultados_local}"
if [ "$MODO_RAPIDO" = 1 ]; then NUM_PAPERS=3 MAX_STEPS=10 MLE_STEPS=2; else NUM_PAPERS=5 MAX_STEPS=20 MLE_STEPS=3; fi

export PATH="$HOME/ollama/bin:$PATH" OPENAI_API_KEY=ollama
export OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_KEEP_ALIVE=-1 OLLAMA_HOST=127.0.0.1:11434
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
  echo "Subindo o Ollama (log em ~/ollama/serve.log)..."
  nohup ollama serve > "$HOME/ollama/serve.log" 2>&1 &
  for _ in $(seq 30); do curl -s http://127.0.0.1:11434/api/tags > /dev/null && break; sleep 1; done
fi
if ! ollama list | grep -q "^$MODELO"; then
  if [ "$MODELO" = qwen3-nothink:8b ]; then
    # qwen3 sem raciocínio (<think>): o bloco vazio que o template insere com think=false passa a ser fixo
    ollama pull qwen3:8b
    ollama show qwen3:8b --modelfile | sed 's/{{ if and \$.IsThinkSet (not \$.Think) -}}/{{ if true -}}/' > /tmp/Modelfile.q3
    grep -q '{{ if true -}}' /tmp/Modelfile.q3 || { echo "o template do qwen3 mudou: veja rodar_local.sh"; exit 1; }
    ollama create qwen3-nothink:8b -f /tmp/Modelfile.q3
  else
    ollama pull "$MODELO"
  fi
fi

cd agent_lab
exec ../.venv/bin/python -u projeto.py "$@" --model "$MODELO" --base_url http://127.0.0.1:11434/v1 \
  --seeds 1 --num_papers "$NUM_PAPERS" --max_steps "$MAX_STEPS" --mle_steps "$MLE_STEPS" --out "$SAIDA" --verbose
