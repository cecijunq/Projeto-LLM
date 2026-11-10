# Classificação das lições (memórias B e C)

58 lições em B e 54 em C (3 seeds cada). Falhas dos 45 projetos de teste (A, B e C somados): 426, agrupadas pela categoria de lição que as atacaria.

Colunas "no prompt": lições que de fato entram no prompt das fases de código (preparação de dados e experimentos), repetindo a consulta dos testes para os 5 temas × 3 seeds.

| Categoria | Lições B | Lições C | No prompt B | No prompt C | Falhas nos testes |
| --- | --- | --- | --- | --- | --- |
| Código e formato | 10 (17%) | 9 (17%) | 28% | 16% | 223 (52%) |
| Ambiente e modelo | 3 (5%) | 6 (11%) | 9% | 16% | 21 (5%) |
| Sintoma sem causa | 14 (24%) | 4 (7%) | 19% | 2% | 115 (27%) |
| Desenho experimental | 12 (21%) | 16 (30%) | 16% | 34% | 63 (15%) |
| Avaliação, análise e escrita | 15 (26%) | 13 (24%) | 15% | 14% | 4 (1%) |
| Dados | 4 (7%) | 6 (11%) | 12% | 19% | 0 (0%) |

- Lições que citam o marcador `</final python code>` (causa frequente de `SyntaxError`): B = 0, C = 3.
- Lições sobre reenviar código que falhou ou erros repetidos: B = 1, C = 4.
- Consultas (tema × seed × fase de código) em que a lição sobre `</final python code>` entra no prompt: C = 12 de 30 (B não tem essa lição).

## Todas as lições

| Versão | Seed | Fase | Categoria | Problema |
| --- | --- | --- | --- | --- |
| B | 0 | data_preparation | Ambiente e modelo | Missing dependencies in data preparation phase caused the project to fail to run. |
| B | 0 | data_preparation | Sintoma sem causa | The project did not reach the report writing phase due to failed data preparation. |
| B | 0 | literature_review | Sintoma sem causa | The literature review was in an invalid format, which may have led to incomplete or incorrect understanding of the topic. |
| B | 0 | data_preparation | Sintoma sem causa | The project did not produce a final report due to the failure to complete the data preparation phase. |
| B | 0 | running_experiments | Código e formato | Syntax errors in the code prevented successful execution of experiments. |
| B | 0 | running_experiments | Ambiente e modelo | Missing dependencies caused import errors in the code. |
| B | 0 | report_writing | Sintoma sem causa | The project did not reach the report writing stage, limiting the ability to communicate findings. |
| B | 0 | literature_review | Sintoma sem causa | The literature review was conducted in a loop, suggesting inefficiency or lack of focus. |
| B | 0 | plan_formulation | Desenho experimental | The experiment did not include a control group or baseline comparison, making it difficult to isolate the effect of gender role prompts. |
| B | 0 | running_experiments | Avaliação, análise e escrita | The experiment reported zero accuracy in both conditions, which may indicate issues with the model's capability rather than the prompts' effect. |
| B | 0 | results_interpretation | Avaliação, análise e escrita | The experiment lacked statistical analysis and error bars, making it hard to assess the reliability of the results. |
| B | 0 | plan_formulation | Avaliação, análise e escrita | The methodology was incomplete, with no description of prompt parsing, model fine-tuning, or evaluation pipeline. |
| B | 0 | running_experiments | Ambiente e modelo | Insufficient model and API validation led to unproductive results with zero accuracy. |
| B | 0 | data_preparation | Dados | Lack of detailed dataset formatting and representation caused unreliable results. |
| B | 0 | literature_review | Avaliação, análise e escrita | Insufficient citation of datasets and models limits the reproducibility and contribution of the work. |
| B | 0 | results_interpretation | Avaliação, análise e escrita | Unanalyzed zero accuracy results lack depth and fail to explore alternative explanations. |
| B | 0 | data_preparation | Dados | Using an incorrect dataset (e.g., 'iris' instead of 'adult') can affect the generalizability of results. |
| B | 0 | results_interpretation | Avaliação, análise e escrita | Lack of statistical analysis (e.g., confidence intervals, hypothesis testing) limits the validity of results. |
| B | 0 | data_preparation | Desenho experimental | Simple label noise simulation may not reflect real-world noise scenarios, limiting the practical relevance of findings. |
| B | 0 | running_experiments | Código e formato | Syntax errors in the code can prevent experiments from running and producing results. |
| B | 1 | data_preparation | Código e formato | Syntax errors in the code due to incorrect string formatting and unterminated string literals. |
| B | 1 | running_experiments | Avaliação, análise e escrita | Insufficient detail in the experimental setup and lack of quantitative analysis. |
| B | 1 | plan_formulation | Desenho experimental | Limited scope due to absence of anchoring bias prompts and lack of systematic variation. |
| B | 1 | results_interpretation | Avaliação, análise e escrita | Qualitative analysis without quantitative measures and limited discussion of limitations and ethical concerns. |
| B | 1 | running_experiments | Código e formato | Syntax errors in code prevented the experiments from running correctly. |
| B | 1 | data_preparation | Dados | The dataset was too small and manually curated, limiting generalizability. |
| B | 1 | plan_formulation | Desenho experimental | The experiment lacked a control group or baseline comparison, making it difficult to assess the impact of the differential diagnosis prompts. |
| B | 1 | data_preparation | Dados | The dataset lacked the required 'problem' column, leading to execution errors and invalid results. |
| B | 1 | running_experiments | Código e formato | The code had an indentation error, which prevented the script from running and producing results. |
| B | 1 | plan_formulation | Desenho experimental | The experimental design lacked justification for prompt formulations, which may have affected the validity of results. |
| B | 1 | results_interpretation | Avaliação, análise e escrita | The project did not present quantitative results or detail the evaluation methodology, which limited its contribution and clarity. |
| B | 1 | data_preparation | Código e formato | Syntax errors in the code prevented the data preparation phase from completing successfully. |
| B | 1 | literature_review | Sintoma sem causa | The literature review phase was not completed, which may have led to an incomplete understanding of existing research on CoT prompting and its effectiveness on math problems. |
| B | 1 | report_writing | Sintoma sem causa | The project did not reach the report writing stage, which is essential for communicating findings and lessons learned. |
| B | 1 | data_preparation | Código e formato | Multiple errors in the data preparation phase suggest poor code quality and lack of testing during development. |
| B | 1 | results_interpretation | Avaliação, análise e escrita | Insufficient statistical analysis was conducted, such as lack of confidence intervals and p-values. |
| B | 1 | plan_formulation | Desenho experimental | No comparison with baseline methods or alternative noise handling techniques was performed. |
| B | 1 | running_experiments | Desenho experimental | Convergence warnings for logistic regression were not addressed in the analysis. |
| B | 1 | data_preparation | Código e formato | Syntax errors in the code prevented the experiments from running correctly. |
| B | 2 | results_interpretation | Avaliação, análise e escrita | Insufficient statistical analysis limits the validity of results |
| B | 2 | plan_formulation | Desenho experimental | Limited generalizability due to reliance on a single model |
| B | 2 | data_preparation | Desenho experimental | Insufficient detail on prompt design and selection |
| B | 2 | report_writing | Avaliação, análise e escrita | Lack of discussion on how findings relate to broader AI bias research |
| B | 2 | data_preparation | Código e formato | Data preparation phase failed due to syntax errors in the code. |
| B | 2 | literature_review | Sintoma sem causa | The literature review was in an invalid format, which may have led to incomplete or incorrect understanding of existing work. |
| B | 2 | report_writing | Sintoma sem causa | The project did not reach the report writing stage, likely due to incomplete execution and lack of documentation. |
| B | 2 | data_preparation | Código e formato | The code for data preparation contained repeated syntax errors, indicating poor code quality and lack of debugging. |
| B | 2 | results_interpretation | Avaliação, análise e escrita | The evaluation criteria were misaligned with the model's output format, leading to zero accuracy scores. |
| B | 2 | results_interpretation | Avaliação, análise e escrita | No statistical analysis was performed on the results, making it impossible to determine the significance of the findings. |
| B | 2 | plan_formulation | Desenho experimental | The experimental design lacked a control condition, making it difficult to isolate the effect of gender role play on model accuracy. |
| B | 2 | results_interpretation | Avaliação, análise e escrita | The raw outputs of the model were not analyzed in detail, making it hard to assess whether valid answers were generated. |
| B | 2 | running_experiments | Desenho experimental | The experiment failed due to timeouts caused by excessive computational load. |
| B | 2 | report_writing | Sintoma sem causa | The project did not complete the report writing phase, which is essential for communicating results and methodology. |
| B | 2 | plan_formulation | Desenho experimental | The experiment design did not account for computational constraints, leading to infeasible execution. |
| B | 2 | running_experiments | Sintoma sem causa | The project did not execute any code successfully, which prevented the collection of empirical results. |
| B | 2 | data_preparation | Sintoma sem causa | Data preparation was not completed successfully, leading to incomplete or incorrect data for experiments. |
| B | 2 | literature_review | Sintoma sem causa | The literature review was not properly formatted, which may have hindered the understanding of existing work and the formulation of the research question. |
| B | 2 | report_writing | Sintoma sem causa | The project did not reach the report writing stage, which is essential for communicating findings and conclusions. |
| C | 0 | data_preparation | Ambiente e modelo | Failed to install required libraries, leading to repeated runtime errors. |
| C | 0 | data_preparation | Ambiente e modelo | Incorrectly used a local language model (e.g., Ollama) without verifying its availability in the environment. |
| C | 0 | data_preparation | Ambiente e modelo | Used an unnecessary library for data preparation, which caused dependency issues. |
| C | 0 | running_experiments | Código e formato | Failed to adapt the approach when encountering repeated errors, leading to a failed experiment. |
| C | 0 | running_experiments | Desenho experimental | The experiment failed to test the research question by comparing a fine-tuned model with random guessing instead of comparing the same LLM with and without differential diagnosis prompts. |
| C | 0 | data_preparation | Código e formato | The code contained invalid syntax due to incorrect use of <final python code> and </final python code> tags, leading to a SyntaxError. |
| C | 0 | running_experiments | Avaliação, análise e escrita | The experiment failed to measure accuracy correctly due to improper comparison of model responses with the ground truth. |
| C | 0 | plan_formulation | Desenho experimental | The experiment design lacked sufficient conditions to test gender role effects, using only one gender role condition. |
| C | 0 | results_interpretation | Avaliação, análise e escrita | The experiment did not validate the accuracy calculation before interpreting results, leading to incorrect conclusions. |
| C | 0 | running_experiments | Código e formato | The code had a syntax error and timeout issues, leading to incomplete or incorrect experimental results. |
| C | 0 | results_interpretation | Avaliação, análise e escrita | The code incorrectly compares the full model response with the complete solution, leading to 0% accuracy. |
| C | 0 | running_experiments | Desenho experimental | The experiment failed to measure the effect of chain-of-thought prompting due to incorrect implementation and data handling. |
| C | 0 | data_preparation | Código e formato | The code had syntax errors and runtime errors due to poor code quality and lack of testing. |
| C | 0 | results_interpretation | Desenho experimental | Missing baseline without label noise makes it impossible to assess the impact of noise on models. |
| C | 0 | data_preparation | Dados | The dataset used in the code does not match the one specified in the plan, leading to incorrect results. |
| C | 0 | running_experiments | Avaliação, análise e escrita | The experiments lacked a fixed random seed and statistical testing, leading to unreliable results. |
| C | 0 | running_experiments | Desenho experimental | The experiment results were inconsistent due to lack of reproducibility and insufficient test size. |
| C | 1 | data_preparation | Dados | The data preparation phase included pre-written model responses instead of actual model outputs, leading to incorrect conclusions. |
| C | 1 | data_preparation | Desenho experimental | The anchoring bias experiment lacked a proper anchor and quantitative metrics, resulting in a qualitative analysis that cannot measure bias effectively. |
| C | 1 | running_experiments | Código e formato | Syntax errors in the code prevented the experiment from running, leading to invalid results. |
| C | 1 | running_experiments | Ambiente e modelo | The experiment failed to properly call the model, leading to incorrect interpretation of results based on pre-written text. |
| C | 1 | data_preparation | Dados | The experiment did not use the MedQA dataset but instead used self-created questions with invented answers, leading to invalid results. |
| C | 1 | running_experiments | Avaliação, análise e escrita | The experiment failed to compute accuracy and only printed responses without proper evaluation metrics. |
| C | 1 | data_preparation | Dados | The dataset used did not contain the expected columns (e.g., 'problem'), leading to runtime errors. |
| C | 1 | running_experiments | Avaliação, análise e escrita | The code attempted to handle errors by skipping rows instead of fixing the root issue, leading to misleading results. |
| C | 1 | results_interpretation | Avaliação, análise e escrita | The experiment was not executed due to errors, yet the report was written as if the results were valid. |
| C | 1 | running_experiments | Código e formato | Code contained invalid syntax due to incorrect use of markdown tags like </final python code>. |
| C | 1 | data_preparation | Dados | Unnecessary train-test split was performed, leading to incorrect data handling and subsequent failures. |
| C | 1 | literature_review | Sintoma sem causa | Literature review was inefficient, spending 20 steps without adding relevant articles. |
| C | 1 | report_writing | Sintoma sem causa | No report was generated, leading to a low score from human researchers. |
| C | 1 | literature_review | Desenho experimental | The literature review failed to identify key issues such as convergence problems in logistic regression and the need for data normalization. |
| C | 1 | plan_formulation | Desenho experimental | The experiment lacked a baseline with 0% label noise and repeated runs with fixed random seeds, leading to unreliable results. |
| C | 1 | data_preparation | Desenho experimental | The logistic regression model did not converge due to lack of data normalization, which was not addressed in the plan or code. |
| C | 1 | results_interpretation | Avaliação, análise e escrita | The results interpretation incorrectly concluded that logistic regression was more affected by label noise, based on incomplete or misleading data. |
| C | 2 | running_experiments | Ambiente e modelo | Calling a non-existent model via the API leads to all responses being error messages instead of actual model outputs. |
| C | 2 | running_experiments | Avaliação, análise e escrita | The code failed to print or check for error responses, leading to incorrect conclusions about model behavior. |
| C | 2 | data_preparation | Desenho experimental | Anchoring prompts lacked a clear anchor, making it impossible to measure the effect of anchoring bias. |
| C | 2 | data_preparation | Desenho experimental | Confirmation bias prompts used binary and potentially biased premises (e.g., 'the Earth is flat' vs. 'the Earth is round'), leading to unreliable results. |
| C | 2 | data_preparation | Código e formato | Data preparation phase failed due to syntax errors in the code. |
| C | 2 | running_experiments | Código e formato | The code submission format was incorrect, leading to failed submissions due to the presence of the </final python code> tag. |
| C | 2 | plan_formulation | Desenho experimental | The experiment design compared the LLM with a rule-based system instead of comparing the same model with and without differential diagnosis prompts. |
| C | 2 | report_writing | Avaliação, análise e escrita | The report lacked a proper submission and failed to include the required results, leading to a low score. |
| C | 2 | results_interpretation | Avaliação, análise e escrita | Comparing full model responses to answers using exact string matching leads to incorrect accuracy calculations. |
| C | 2 | data_preparation | Desenho experimental | Using trivial math questions fails to detect any gender role play effects on model accuracy. |
| C | 2 | plan_formulation | Desenho experimental | The experiment lacked a control condition without gender role prompts, making it difficult to interpret results. |
| C | 2 | report_writing | Avaliação, análise e escrita | The report was written despite the results being invalid due to an error in evaluation. |
| C | 2 | running_experiments | Desenho experimental | The code made 400 model queries (200 questions × 2 conditions), leading to timeouts and exceeding the allowed time limit. |
| C | 2 | running_experiments | Desenho experimental | The CoT prompting condition did not explicitly ask for step-by-step reasoning, making the comparison with the full solution ineffective. |
| C | 2 | results_interpretation | Avaliação, análise e escrita | The comparison between the model's answer and the full solution of GSM8K was flawed, leading to an incorrect conclusion of 0% accuracy. |
| C | 2 | report_writing | Sintoma sem causa | The report was not generated, which prevented the dissemination of findings and conclusions. |
| C | 2 | data_preparation | Dados | The dataset used in the experiment was not aligned with the plan, leading to irrelevance and confusion. |
| C | 2 | data_preparation | Ambiente e modelo | Label noise was introduced during data preparation, but the code lacked necessary imports, leading to runtime errors. |
| C | 2 | running_experiments | Código e formato | The code was repeatedly submitted without changes, leading to inefficiency and failure to resolve errors. |
| C | 2 | report_writing | Sintoma sem causa | The report was missing, which prevented proper evaluation of the results and conclusions. |
