"""Versão D (exploratória, fora da comparação A/B/C): correções do sistema tiradas dos erros vistos no TREINO.

Só valem com AGENTLAB_CORRECOES=1 (projeto.py liga para 'teste --versao D'). Desligadas, as funções devolvem ""
e A, B e C rodam exatamente com as instruções de sempre.
"""
import os
import re

LLM_CALL = ("To query the language model in code, use exactly: from openai import OpenAI; "
            "client = OpenAI(base_url=os.environ.get('AGENTLAB_LLM_BASE_URL'), api_key='x'); "
            "r = client.chat.completions.create(model=os.environ['AGENTLAB_LLM_MODEL'], messages=[...], "
            "max_tokens=256, temperature=0). The 'ollama' Python package is NOT installed: never import ollama. "
            "Always pass model=os.environ['AGENTLAB_LLM_MODEL']: other names such as 'llama3' are not installed, and "
            "do not call the Ollama HTTP API with requests.")  # (seed2/t0: 'llama3' via requests, todas as respostas erro)

# entre parênteses, o projeto do treino de onde veio cada correção
PLAN = ("For a topic of the form 'does X (a prompt, a role, an order, a reasoning style) change the accuracy of a "
        "language model?', compare the SAME local model with and without X, on the SAME questions, with the same "
        "metric, and include every condition the topic names (e.g. both a man and a woman for gender roles). Do not "
        "train any model and do not compare against random guessing. "  # (seed0/t1, seed0/t2)
        "For a topic of the form 'does Y hurt model M more than model N?', include the clean condition (Y = 0) and "
        "compare the DROP of each model from its own clean accuracy, not the accuracies alone. "  # (seed0/t4)
        "Use a test set of at least 200 examples. "  # (seed0/t4: 30 exemplos de teste)
        "For a cognitive bias, compare paired versions of the same item and measure the difference with a number: "
        "anchoring = the same estimation question with a high anchor, a low anchor and no anchor (measure how far "
        "the model's numeric answer moves toward the anchor); confirmation = the same claim framed for and against "
        "(measure how often the model agrees with each). "  # (seed1/t0: 'ancoragem' sem âncora, só análise qualitativa)
        "A prompting experiment needs no training set: evaluate directly on the test split. "  # (seed1/t3)
        "A chain-of-thought condition must explicitly ask the model to think step by step; a gender-role condition "
        "needs a no-role control besides the male and female roles. "  # (seed2/t3: 'CoT' era só 'Q: ... A:'; seed2/t2)
        "Avoid trivial questions (e.g. 'What is 2 + 2?'): every condition would score near 100%.")  # (seed2/t2)

LIT = ("After each SUMMARY search, add at least one relevant paper with ADD_PAPER before searching again, and never "
       "repeat a search query.")  # (seed1/t3, seed1/t4: 20 passos buscando sem adicionar artigos)

DATA_PHD = ("As soon as the ML engineer's code runs and loads the data the plan needs, submit it with SUBMIT_CODE; do "
            "not keep repeating that you will write the code.")  # (seed2/t1: ~15 turnos repetindo a mesma mensagem)

DATA = ("Data preparation only loads and formats the data (questions and gold answers): do not query the language "
        "model, train models or compute results here. "  # (seed0/t0, seed0/t4)
        "Use exactly the dataset in the plan; if it cannot be loaded, say so instead of silently switching to "
        "another one. "  # (seed0/t4: Adult no plano, Iris no código)
        "If the topic names a benchmark (MedQA, GSM8K, SST-2...), load that benchmark from the verified list "
        "(MedQA: GBaker/MedQA-USMLE-4-options); never replace it with hand-written questions. "  # (seed1/t1)
        "If the plan names a scikit-learn or torchvision dataset (e.g. fetch_california_housing, load_breast_cancer), "
        "load it with that library instead of searching HuggingFace, and never switch to an unrelated dataset just "
        "because it is in the list; the label must be a meaningful target column, not an id or a count. "
        # (seed2/t4: California Housing no plano, 'car-classification' com alvo 'num' no código)
        "Label noise, augmentation and other experimental manipulations belong to the experiment, not to data "
        "preparation. "  # (seed2/t4)
        "Every code cell must include all its imports (e.g. import numpy as np). "  # (seed2/t4: NameError np)
        "For math word problems use openai/gsm8k (config 'main'). The dataset must have one column with the question "
        "and one with the gold answer: print df.columns and one full example, and later code must use exactly those "
        "column names. "  # (seed1/t2: dataset sem coluna de enunciado; código supôs 'problem'/'solution')
        "The data holds only inputs and gold labels, never model responses: the responses must come from the model "
        "in the experiment. "  # (seed1/t0: o agente escreveu as 'respostas do modelo' nos dados)
        "Write hand-written examples as a Python list of dicts, one per line, using double quotes and escaping any "
        "quote inside the text. "  # (seed1/t0: SyntaxError de aspas)
        "dataset.train_test_split(...) returns a DatasetDict: use split['train'] and split['test'], never "
        "'a, b = dataset.train_test_split(...)' (that gives the strings 'train' and 'test'). "  # (seed1/t3)
        "In GSM8K the gold answer is the number after '####': answer.split('####')[-1].strip(). " + LLM_CALL)  # (seed0/t3)

INTERP = ("If an accuracy is 0% or 100%, or exactly the same in all conditions, treat it as a bug in the evaluation, "
          "not as a finding: say so explicitly and draw no conclusion about the model from it. "  # (seed0/t2, seed0/t3)
          "Claim that one condition is better or more robust only if a significance test supports it and the "
          "difference is larger than the variation between runs. "  # (seed0/t4)
          "Base every claim only on numbers printed in the results; never describe what the model 'would likely' do. "
          "If no metric was computed, say that the experiment produced no result. "  # (seed1/t1)
          "Every number you cite must appear literally in the results: never invent a baseline that was not run. "
          "Before concluding, check the direction: the model whose metric drops more is the one hurt more; if the "
          "numbers go up and down across levels (not monotonic), say the result is noisy. "  # (seed1/t4)
          "If the results only say that files were saved, or show error messages instead of model answers, the "
          "experiment produced no result: say so and claim nothing about the model.")  # (seed2/t0)

REFINO = (" If the experiment produced no valid result (no metric, 0% or identical accuracy in all conditions, error "
          "messages instead of model answers) or the interpretation says the evaluation is broken, reply REVISIT "
          "running_experiments instead of FINISH.")  # (seed2/t2: a interpretação achou o bug e o relatório saiu igual)

TAREFA = {("phd", "literature_review"): LIT, ("phd", "plan_formulation"): PLAN, ("postdoc", "plan_formulation"): PLAN,
          ("phd", "data_preparation"): DATA_PHD + " " + DATA, ("mle", "data_preparation"): DATA,
          ("phd", "results_interpretation"): INTERP, ("postdoc", "results_interpretation"): INTERP}

AVALIACAO = ("\nWhen evaluating a language model's answers: state the answer format in the prompt (e.g. 'Think step by "
             "step and end with: Answer: <number>'), use max_tokens=256 when it reasons, extract the final answer from "
             "the response (e.g. the last number, with a regex) and compare it with the gold answer; never compare the "
             "full response with ==. Print 3 examples (model response, extracted answer, gold answer) before each "
             "accuracy. "  # (seed0/t2, seed0/t3)
             "Every model response you print or score must come from a real call to the model in this code; never "
             "print or score responses stored in the data. "  # (seed1/t0)
             "Printing responses is not a result: for each condition compute and print the metric as "
             "'accuracy (<condition>): <correct>/<total> = <value>'. For multiple-choice questions, list the options "
             "with letters, ask for the letter only and compare it with the gold letter. "  # (seed1/t1)
             "Each model call takes seconds: evaluate at most 50 questions per condition (df = df.head(50)), or the "
             "code is stopped at the time limit. "  # (seed2/t3: 200 questões x 2 condições, 5 timeouts)
             "Print the results on screen: saving them to a file is not enough. "  # (seed2/t0)
             "Do not hide errors with try/except or 'if ...: continue': if the data is not as expected, let the code "
             "fail so it can be fixed. "  # (seed1/t2)
             "Fix all random seeds at the start (random.seed, np.random.seed, torch.manual_seed) so results are "
             "reproducible. Repeat each condition at least 5 times and print mean, standard deviation and a paired "
             "significance test (scipy.stats.ttest_rel or wilcoxon). "  # (seed0/t4)
             "To add label noise, replace each chosen label with a DIFFERENT class (a random permutation often keeps "
             "the same label). "  # (seed0/t4)
             "Scale the features (StandardScaler) before logistic regression or other linear models, so they "
             "converge. "  # (seed1/t4: ConvergenceWarning em todos os treinos)
             "Write code lines without the 'N |' line numbers shown in 'Current code'. "  # (seed1/t4)
             "Inside the code block write only Python: never '<final python code>' or "
             "'</final python code>'. AdamW comes from torch.optim, not from transformers. " + LLM_CALL)  # (seed0/t1)

REPARO = (" If a function that queries the language model is missing (e.g. NameError: run_prompt), define it with "
          "the client call below; never replace the model call with stored or hard-coded responses. "
          # (seed1/t0: o reparo trocou a chamada ao modelo por example['response'])
          "Fix the cause of the error; never hide it with try/except, 'if ...: continue' or by skipping the data "
          "(e.g. for a KeyError on a column, use the real column name shown in the data). "
          + LLM_CALL)  # (seed1/t2: o reparo pulou as 200 linhas e o modelo nunca foi consultado)

# linha de resultado: nome de métrica seguido de um número (seed1/t1 e seed1/t2 passaram na checagem antiga
# só com listas numeradas e com a tabela df.head())
# o número pode vir até 4 linhas depois do nome (tabela do pandas, seed0/t4)
_METRICA = re.compile(r"\b(accuracy|acc|f1|precision|recall|error rate|loss|score|mean|p-value|p_value|shift)\b"
                      r"(?:[^\n]*\n){0,4}?[^\n]{0,80}?(\d\.\d|\d\s*/\s*\d|\d\s*%)", re.I)
SEM_METRICA = ("[ERRO] The code ran but printed no metric. For each condition print a line like "
               "'accuracy (<condition>): <correct>/<total> = <value>'.\n")

_FECHA = re.compile(r"</[a-z][a-z0-9 _/-]*>")  # '</final python code>' copiado do exemplo de comando


def ativo() -> bool:
    return os.environ.get("AGENTLAB_CORRECOES") == "1"


def tarefa(kind: str, phase: str) -> str:
    return " " + TAREFA[(kind, phase)] if ativo() and (kind, phase) in TAREFA else ""


def avaliacao() -> str:
    return AVALIACAO if ativo() else ""


def reparo() -> str:
    return REPARO if ativo() else ""


REPETIDO = ("[WARNING] You sent exactly the same code that already failed. Read the error below and change the code "
            "(e.g. for NameError add the missing import or definition).\n")  # (seed2/t4: ~14 vezes o mesmo código)


def repetido(code: str, last_failed: str) -> str:
    return REPETIDO if ativo() and code.strip() == last_failed.strip() else ""


def refino() -> str:
    return REFINO if ativo() else ""


def sem_metrica(exp_out: str) -> bool:
    return ativo() and not _METRICA.search(exp_out)


_NUMERO = re.compile(r"^\s*\d+ \|", re.M)  # 'N |' da listagem 'Current code' copiado para o código (seed1/t4)


def limpa(content: str) -> str:
    if not ativo():
        return content
    head, _, last = content.rstrip().rpartition("\n")
    if _FECHA.fullmatch(last.strip()):
        content = head.rstrip()
    return _NUMERO.sub(lambda m: m.group(0)[:len(m.group(0)) - len(m.group(0).lstrip())], content)  # mantém a indentação
