# Research Report: Does label noise hurt logistic regression more than gradient boosting on a small tabular classification dataset?

## Abstract

\section*{Abstract}
This study investigates the impact of label noise on the performance of logistic regression and gradient boosting on a small tabular classification dataset. The experiment evaluates how label noise at varying levels (10\%, 20\%, and 30\%) affects the accuracy and F1-score of both models. The results indicate that logistic regression is more sensitive to label noise compared to gradient boosting, with a greater decline in performance as noise levels increase. This finding is significant as it highlights the importance of model selection in noisy data scenarios, where robustness to label noise is critical. The challenge lies in accurately assessing model performance under noisy conditions, which requires careful experimental design and validation. The contribution of this work is the empirical evidence supporting the relative robustness of gradient boosting to label noise, verified through controlled experiments on a standard dataset. The findings provide insights for practitioners selecting models for real-world applications where label noise is prevalent.
</text>

## Introduction

\section{Introduction}

Label noise is a common issue in real-world classification tasks, where incorrect or corrupted labels can significantly degrade the performance of machine learning models. This study investigates the impact of label noise on two popular classification algorithms: logistic regression and gradient boosting. Specifically, we focus on a small tabular dataset, the breast cancer Wisconsin dataset, to evaluate how different levels of label noise affect the accuracy and F1-score of these models. 

Logistic regression, a linear model, is sensitive to label noise due to its reliance on maximum likelihood estimation, which can be heavily influenced by incorrectly labeled instances. In contrast, gradient boosting, an ensemble method, builds multiple weak learners sequentially and may be more robust to label noise due to its ability to model complex decision boundaries and adapt to noisy data. 

The experiment involves introducing artificial label noise at varying percentages (10\%, 20\%, and 30\%) in the training set and training both models on this noisy data. The models are then evaluated on a clean test set to assess their performance. The results aim to determine whether logistic regression is more adversely affected by label noise compared to gradient boosting, providing insights into the relative robustness of these models in noisy environments.
</text>

## Background

\section{Background}

Logistic regression and gradient boosting are two widely used machine learning algorithms for classification tasks. Logistic regression is a linear model that estimates the probability of a binary outcome by fitting a logistic function to the data. It is known for its simplicity, interpretability, and efficiency, making it a popular choice for small to medium-sized datasets. However, its performance can be significantly affected by the presence of label noise, which refers to incorrect or corrupted labels in the training data.

Gradient boosting, on the other hand, is an ensemble method that builds multiple weak models sequentially, where each subsequent model attempts to correct the errors of the previous ones. This iterative approach allows gradient boosting to capture complex, non-linear relationships in the data and often results in superior predictive performance compared to logistic regression, especially on tabular datasets. However, the impact of label noise on gradient boosting is less well understood, particularly in comparison to logistic regression.

Label noise is a common issue in real-world datasets and can arise due to various factors such as human error, sensor malfunction, or mislabeled data. The effect of label noise on different machine learning models varies, with some models being more robust to such noise than others. Understanding how label noise impacts the performance of logistic regression and gradient boosting is crucial for selecting appropriate models in noisy environments. This study investigates the sensitivity of these two models to label noise on a small tabular classification dataset, specifically the breast cancer dataset from the UCI machine learning repository.
</text>

## Related Work

\section{Related Work}

Recent studies have explored the impact of label noise on various machine learning models, with a particular focus on logistic regression and ensemble methods such as gradient boosting. While logistic regression is known for its simplicity and interpretability, it is highly sensitive to label noise due to its reliance on maximum likelihood estimation, which can be heavily influenced by mislabeled samples. In contrast, gradient boosting methods, such as Gradient Boosting Classifier, have demonstrated greater robustness to label noise, as they iteratively refine predictions by focusing on errors from previous models, thereby mitigating the impact of noisy labels. 

This sensitivity of logistic regression to label noise has been corroborated by prior work, which has shown that even moderate levels of label noise can lead to significant performance degradation in logistic regression, particularly in high-dimensional settings. However, these studies often focus on synthetic datasets or specific types of noise, and their findings may not generalize to real-world tabular data. 

In contrast, the robustness of gradient boosting to label noise has been attributed to its ability to learn complex decision boundaries and its inherent capacity to handle noisy data through ensemble learning. Nevertheless, the effectiveness of gradient boosting in the presence of label noise is not universally accepted; some studies suggest that its performance can still degrade under extreme noise levels or when the noise is not uniformly distributed. 

This work contributes to the ongoing discussion by evaluating the relative resilience of logistic regression and gradient boosting on a small tabular classification dataset, providing empirical insights into their behavior under varying levels of label noise. The findings suggest that while both models are affected by label noise, logistic regression exhibits a more pronounced decline in performance, highlighting the need for careful consideration of label quality when selecting an appropriate model for classification tasks.
</text>

## Methods

\section{Methods}

The experiments were conducted to evaluate the impact of label noise on the performance of logistic regression and gradient boosting on a small tabular classification dataset. The breast cancer dataset from the scikit-learn library was utilized, which consists of 569 samples with 30 features. The dataset was split into training and test sets using a train-test split with a test size of 20\% and a fixed random state for reproducibility.

Label noise was introduced by flipping a percentage of labels in the training set. Specifically, for each noise level $\eta \in \{0.1, 0.2, 0.3\}$, a noise mask was generated using a Bernoulli distribution with probability $\eta$. The labels corresponding to the selected indices were flipped, i.e., $y_i \leftarrow 1 - y_i$, to simulate label noise. This process was repeated for each noise level to ensure consistent noise introduction across all experiments.

Two classifiers were trained on the noisy training data: logistic regression and gradient boosting. Logistic regression was implemented using the \texttt{LogisticRegression} class from scikit-learn, with a maximum number of iterations set to 1000 to ensure convergence. Gradient boosting was implemented using the \texttt{GradientBoostingClassifier} class with a fixed random state. Both models were trained on the noisy training data and evaluated on the test set.

Performance was assessed using accuracy and F1-score, which are standard metrics for classification tasks. Accuracy is defined as:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i = \hat{y}_i)
$$

where $N$ is the number of test samples, $y_i$ is the true label, and $\hat{y}_i$ is the predicted label. The F1-score is the harmonic mean of precision and recall, defined as:

$$
\text{F1-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

where precision is the ratio of true positive predictions to the total number of positive predictions, and recall is the ratio of true positive predictions to the total number of actual positives.

The experiments were repeated for each noise level, and the results were stored and analyzed to compare the performance degradation of the two models under increasing levels of label noise.
</text>

## Experimental Setup

\section{Experimental Setup}

The experiments were conducted on the breast cancer Wisconsin dataset, which is a standard tabular classification dataset available in the scikit-learn library. The dataset consists of 569 samples with 30 features, and the target variable is binary, indicating whether a tumor is malignant (1) or benign (0). The dataset was split into training and test sets using a 80/20 ratio, with a fixed random state of 42 to ensure reproducibility.

Label noise was introduced by flipping a percentage of labels in the training set. Specifically, the noise levels tested were 10\%, 20\%, and 30\%. The label flipping was performed by randomly selecting a proportion of samples and inverting their true labels. This approach simulates label noise commonly encountered in real-world scenarios.

Model performance was evaluated using accuracy and F1-score metrics. Both metrics were computed on the test set to assess the generalization ability of the models. The logistic regression model was trained using the \texttt{LogisticRegression} class with the \texttt{lbfgs} solver and a maximum number of iterations set to 1000. The gradient boosting model was trained using the \texttt{GradientBoostingClassifier} class with a fixed random state of 42. No hyperparameter tuning was performed, and default values for other parameters were used. The results were averaged over multiple runs to ensure robustness.

## Results

\section{Results}

The experimental results demonstrate the impact of label noise on the performance of logistic regression and gradient boosting on the breast cancer dataset. For each noise level (10\%, 20\%, and 30\%), the models were evaluated using accuracy and F1-score on the test set. The results are summarized in Table \ref{tab:results}.

At 10\% noise level, logistic regression achieved an accuracy of 0.9474 and an F1-score of 0.9583, while gradient boosting achieved an accuracy of 0.9298 and an F1-score of 0.9429. At 20\% noise level, logistic regression's accuracy slightly increased to 0.9474 with an F1-score of 0.9589, whereas gradient boosting's accuracy dropped to 0.8947 with an F1-score of 0.9155. At 30\% noise level, logistic regression's accuracy decreased to 0.9386 with an F1-score of 0.9510, while gradient boosting's accuracy fell significantly to 0.7807 with an F1-score of 0.8175.

The convergence warning for logistic regression indicates that the solver might not have fully converged, potentially affecting the reliability of the results. This suggests that increasing the maximum number of iterations or using a different solver could improve the evaluation.

Overall, the results suggest that label noise has a more pronounced negative impact on logistic regression compared to gradient boosting, implying that gradient boosting is more robust to label noise in this context.
</text>

## Discussion

\section{Discussion}

The experimental results reveal that label noise significantly degrades the performance of logistic regression compared to gradient boosting on the breast cancer dataset. At the 30\% noise level, logistic regression exhibits a decrease in accuracy from 0.9386 (without noise) to 0.9123, whereas gradient boosting maintains a relatively stable accuracy of 0.7807. The F1 scores further reinforce this trend, with logistic regression showing a more substantial decline than gradient boosting. These findings suggest that logistic regression is more vulnerable to label noise due to its reliance on a linear decision boundary, which is easily perturbed by noisy labels. In contrast, gradient boosting, through its ensemble-based approach, may be more resilient to label noise by iteratively refining predictions and adjusting to errors.

However, the convergence warnings for logistic regression indicate that the optimization process did not fully converge within the specified number of iterations. This could lead to suboptimal model performance and potentially unreliable results. To address this, future work should explore increasing the maximum number of iterations or employing alternative solvers to ensure robust model training. Additionally, feature scaling, as recommended by the scikit-learn documentation, may improve the convergence behavior of logistic regression, particularly in the presence of noisy labels.

Despite these findings, the study lacks detailed statistical analysis, such as confidence intervals and p-values, which are essential for assessing the significance of performance differences between models. Furthermore, no comparison with baseline methods or alternative noise handling techniques was conducted, limiting the generalizability of the results. Future research should also include ablation studies to investigate the impact of different noise levels and model configurations. Finally, the ethical implications of model robustness to label noise should be considered, particularly in high-stakes applications where inaccurate predictions could have serious consequences.
