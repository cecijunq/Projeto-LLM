# Memória C (Memória + humano) · seed 0


## seed0/t0 · Do language models exhibit cognitive biases, such as confirmation bias or anchoring bias?
- **data_preparation** · Failed to install or import required libraries, leading to repeated runtime errors. → Check if required libraries are installed in the environment before running the code. If a library is not available, consider using an alternative or install the missing package. (importância 0.9)
- **data_preparation** · Incorrectly used a local language model (e.g., Ollama) without verifying its availability in the environment. → Verify the availability of the chosen model and its dependencies in the environment. Prefer models that are already supported or ensure the environment is properly configured. (importância 0.8)
- **data_preparation** · Used a language model during data preparation, which is unnecessary and caused errors. → Separate the data preparation phase from model usage. Only load and preprocess data during this phase, and use the model during the experiment execution phase. (importância 0.7)
- **running_experiments** · Repeated the same approach without adapting when errors occurred, leading to multiple failures. → When an error occurs, evaluate whether the approach needs to be changed. If the same error persists, consider alternative methods or tools. (importância 0.6)

## seed0/t1 · Do language models improve accuracy on MedQA when asked to perform differential diagnosis?


## seed0/t2 · Does gender role play affect the accuracy of language models on answering math questions?


## seed0/t3 · Does chain-of-thought prompting improve the accuracy of a small language model on grade-school math (GSM8K)?


## seed0/t4 · Does label noise hurt logistic regression more than gradient boosting on a small tabular classification dataset?
- **results_interpretation** · Missing baseline without label noise makes it impossible to assess the impact of noise on models. → Include a baseline condition with 0% label noise and compare the performance drop of each model relative to their clean accuracy. (importância 0.9)
- **data_preparation** · The wrong dataset was used, leading to misleading results and invalid comparisons. → Ensure the dataset used in the code matches the one specified in the plan. Verify the dataset name, size, and structure before proceeding with experiments. (importância 0.9)
- **running_experiments** · Lack of fixed random seed and repeated experiments leads to inconsistent results. → Fix the random seed using np.random.seed() or similar functions and repeat the experiments at least 5 times to ensure reproducibility and statistical validity. (importância 0.8)
- **results_interpretation** · No statistical test was performed to validate the conclusion about model robustness. → Perform a paired statistical test (e.g., paired t-test) to compare the performance of models across noise levels and validate the conclusion about robustness. (importância 0.8)
