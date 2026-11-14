# Memória C (Memória + humano) · seed 2


## seed2/t0 · Do language models exhibit cognitive biases, such as confirmation bias or anchoring bias?
- **running_experiments** · Calling a non-existent model via the API leads to incorrect results and no meaningful data collection. → Use a model that is available and properly configured, such as the OpenAI client with the correct model name specified in the environment variable. (importância 0.9)
- **running_experiments** · The code failed to check for and handle errors in API responses, leading to invalid conclusions. → Implement error checking in the code to ensure responses are valid and not error messages. Print or log results for verification. (importância 0.9)
- **data_preparation** · The anchoring prompts lacked a proper anchor, making it impossible to measure the effect of anchoring bias. → Design anchoring prompts with distinct anchor values and include a control condition without an anchor to measure the effect accurately. (importância 0.8)
- **results_interpretation** · The interpretation of results was based on insufficient or invalid data, leading to incorrect conclusions. → Ensure that the data collected is valid and contains measurable metrics before interpreting results. Include quantitative analysis to support conclusions. (importância 0.8)

## seed2/t1 · Do language models improve accuracy on MedQA when asked to perform differential diagnosis?


## seed2/t2 · Does gender role play affect the accuracy of language models on answering math questions?
- **results_interpretation** · Comparing full model responses to answers using exact string matching leads to incorrect accuracy calculations. → Extract numerical answers from model responses and compare them numerically instead of using exact string matching. (importância 0.9)
- **data_preparation** · Using trivial math questions fails to detect any meaningful differences in model performance based on gender role prompts. → Use more complex math questions from datasets like GSM8K to ensure sufficient difficulty and variability in responses. (importância 0.8)
- **plan_formulation** · The experiment lacked a control condition without gender role prompts, making it difficult to interpret results. → Include a control condition where the model is not prompted with a gender role to isolate the effect of the prompt. (importância 0.8)
- **report_writing** · The report was written despite the interpretation identifying an error in the evaluation process. → Only proceed to report writing after confirming that the evaluation process is accurate and free of errors. (importância 0.7)

## seed2/t3 · Does chain-of-thought prompting improve the accuracy of a small language model on grade-school math (GSM8K)?
- **running_experiments** · The code made 400 model queries (200 questions × 2 conditions), leading to timeouts and exceeding the allowed time limit. → Limit the number of model queries to a maximum of 50 per condition by using a subset of the data, such as df.head(50). (importância 0.9)
- **running_experiments** · The CoT prompting condition did not explicitly ask for step-by-step reasoning, making the comparison with the full solution of GSM8K ineffective. → Explicitly request step-by-step reasoning in the CoT condition, such as by including the instruction 'Think step by step and end with: Answer: <number>'. (importância 0.9)
- **results_interpretation** · The evaluation compared the full text response with the exact solution, which would result in 0% accuracy even if the model provided the correct answer. → Extract only the numerical answer from the model's response and compare it with the correct number from the GSM8K dataset to ensure accurate evaluation. (importância 0.8)
- **report_writing** · The report was not generated, which prevented the researchers from reviewing the findings and conclusions. → Ensure that the report-writing phase is completed by including a structured report that summarizes the methodology, results, and conclusions. (importância 0.8)

## seed2/t4 · Does label noise hurt logistic regression more than gradient boosting on a small tabular classification dataset?
- **data_preparation** · The dataset used in the experiment was not aligned with the plan, leading to irrelevant data and incorrect target labels. → Ensure that the dataset used in the experiment exactly matches the one described in the plan. Verify that the target variable is correctly identified and used as the label. (importância 0.9)
- **data_preparation** · Label noise was introduced during data preparation, but the code lacked necessary imports, leading to runtime errors. → Include all required imports at the beginning of each code section. Validate that the code can run without errors before submitting it for execution. (importância 0.8)
- **running_experiments** · The code was repeatedly submitted without changes, leading to the same errors persisting. → Review error messages carefully and make necessary adjustments before resubmitting code. Avoid submitting the same code multiple times without modifications. (importância 0.8)
- **report_writing** · No report was submitted, making it impossible to assess the findings or methodology. → Ensure that a complete report is generated and submitted as part of the project, including methodology, results, and conclusions. (importância 0.9)
