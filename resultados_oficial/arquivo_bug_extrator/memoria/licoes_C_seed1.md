# Memória C (Memória + humano) · seed 1


## seed1/t0 · Do language models exhibit cognitive biases, such as confirmation bias or anchoring bias?
- **data_preparation** · The data preparation phase included hardcoded model responses instead of actual model outputs, leading to incorrect conclusions. → Ensure that the data only contains prompts, and that the model's responses are obtained through a real API call during the experiment. Avoid hardcoding any responses. (importância 0.9)
- **data_preparation** · The anchoring bias experiment lacked a proper anchor and quantitative measurement, resulting in a qualitative analysis that cannot determine bias. → For anchoring bias, use the same estimation question with high anchor, low anchor, and no anchor, and measure how much the numerical response shifts from the anchor. (importância 0.8)
- **data_preparation** · The code had multiple syntax errors due to incorrect string formatting and unterminated string literals. → Carefully check the syntax of string literals and ensure that all strings are properly terminated. Use code linters or syntax checkers to catch such errors early. (importância 0.8)
- **running_experiments** · The experiment failed to properly call the model, leading to the use of the agent's own text as the model's response, which invalidated the results. → Define a function to call the model using the appropriate API (e.g., OpenAI) and ensure that the model's responses are retrieved through this function, not hardcoded. (importância 0.9)

## seed1/t1 · Do language models improve accuracy on MedQA when asked to perform differential diagnosis?


## seed1/t2 · Does gender role play affect the accuracy of language models on answering math questions?


## seed1/t3 · Does chain-of-thought prompting improve the accuracy of a small language model on grade-school math (GSM8K)?
- **running_experiments** · Code contained invalid syntax due to incorrect use of markdown tags like </final python code>. → Ensure that only valid Python code is written within code blocks, avoiding any markdown syntax or tags. (importância 0.9)
- **data_preparation** · Unnecessary train-test split was performed, leading to incorrect data handling and subsequent failures. → Use the built-in test split from the dataset (e.g., load_dataset('openai/gsm8k', 'main', split='test[:200]')) instead of creating an unnecessary split. If splitting is required, use the train_test_split method correctly. (importância 0.8)
- **literature_review** · Literature review was conducted without adding relevant articles, leading to an incomplete review. → After each search step in the literature review, add at least one relevant article to ensure comprehensive coverage of the topic. (importância 0.7)
- **report_writing** · No report was generated, leading to a lack of documentation and understanding of the project. → Ensure that a report is created and properly documented, summarizing the methodology, findings, and conclusions of the project. (importância 0.8)

## seed1/t4 · Does label noise hurt logistic regression more than gradient boosting on a small tabular classification dataset?
- **literature_review** · The literature review failed to identify key issues such as the need for a baseline (0% noise) and proper normalization for logistic regression. → Conduct a thorough literature review that includes baseline experiments and preprocessing techniques relevant to the models being used. (importância 0.9)
- **plan_formulation** · The experiment lacked a baseline (0% noise) and repeated runs with fixed random seeds, leading to unreliable comparisons. → Include a 0% noise baseline and repeat experiments with fixed random seeds to ensure consistent and reliable results. (importância 0.8)
- **data_preparation** · The data was not normalized before training logistic regression, leading to convergence issues and poor performance. → Normalize the data using techniques like StandardScaler before training logistic regression to ensure convergence and better performance. (importância 0.9)
- **running_experiments** · Syntax errors in the code prevented the experiments from running correctly, leading to invalid results. → Thoroughly test the code for syntax errors before running experiments to ensure all parts of the pipeline are functional. (importância 0.8)
