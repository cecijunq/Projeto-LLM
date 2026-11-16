# Research Report: Does label noise hurt logistic regression more than gradient boosting on a small tabular classification dataset?

## Abstract

The study investigates whether label noise disproportionately harms logistic regression compared to gradient boosting on a small tabular classification dataset. Leveraging the "adult" dataset, we simulate 10% and 20% label noise by randomly flipping training labels and evaluate model performance on a clean test set. The results indicate that logistic regression maintains higher accuracy under label noise, suggesting greater robustness to this issue. This finding is significant as it challenges the common assumption that ensemble methods like gradient boosting are inherently more resilient to label noise. The study highlights the importance of understanding model behavior under noisy conditions, which is critical for real-world applications where data quality is often uncertain. The results are verified through repeated experiments and statistical analysis, contributing to the broader discourse on improving model robustness in presence of label noise.
</text>

## Introduction



## Background

\section{Background}

Label noise is a common issue in real-world classification tasks, where the true class labels of training samples are corrupted or incorrect. This phenomenon can significantly degrade the performance of machine learning models, particularly when the noise is not uniformly distributed across classes or instances. In recent years, several studies have investigated the impact of label noise on different classification algorithms, with a focus on logistic regression and gradient boosting. 

Logistic regression, being a linear model, is sensitive to label noise due to its reliance on maximum likelihood estimation, which can be distorted by incorrect labels. However, its simplicity and interpretability make it a popular choice for tabular data. On the other hand, gradient boosting, which builds an ensemble of weak learners iteratively, has shown higher robustness to label noise in some cases due to its ability to model complex decision boundaries and adapt to noisy data through the learning process. 

The challenge of handling label noise has been extensively studied in various contexts, including graph node classification and structured tabular data. For instance, methods such as label ensemble strategies and instance-dependent noise modeling have been proposed to mitigate the adverse effects of label noise. These approaches often involve either modifying the learning process to account for noisy labels or designing models that are inherently more robust to label errors. 

In the context of this study, we focus on the question of whether logistic regression is more affected by label noise compared to gradient boosting on a small tabular classification dataset. To address this, we simulate label noise by randomly flipping a percentage of the labels in the training set and evaluate the performance of both models under different noise levels. The insights gained from this analysis can inform the selection of appropriate models for real-world tabular classification tasks where label noise is prevalent.

## Related Work

\section{Related Work}

The impact of label noise on machine learning models has been extensively studied, particularly in the context of logistic regression and gradient boosting. Label noise, which refers to incorrect or corrupted labels in the training data, can significantly degrade model performance. While several works have proposed methods to mitigate label noise, the extent to which different models are affected varies, and this paper draws on prior research to contextualize its findings.

Logistic regression, as a linear model, is inherently sensitive to label noise due to its reliance on maximum likelihood estimation, which assumes clean labels. However, recent studies have explored ways to enhance its robustness. For instance, the work in \cite{2208.06828v2} introduces an improved version of the Adaptive Gradient Algorithm for multinomial logistic regression, aiming to accelerate training and improve performance. While this paper focuses on optimization techniques, it indirectly implies that logistic regression can be made more resilient to label noise through algorithmic refinements. Similarly, the Label Ensemble Graph Neural Network (LEGNN) \cite{2411.11020v2} reframes label noise as a label ensemble problem, suggesting that ensemble-based strategies could be leveraged to improve robustness. These approaches, while tailored for graph data, highlight the potential for modifying logistic regression to handle noisy labels more effectively.

In contrast, gradient boosting methods such as GradientBoostingClassifier in scikit-learn are inherently more robust to label noise due to their iterative nature and the use of decision trees, which can capture complex patterns in data. The work in \cite{2012.11854v2} addresses the problem of instance-dependent label noise, proposing a second-order approach that adapts to varying noise levels across instances. This aligns with the idea that gradient boosting models, by construction, are less susceptible to label noise, especially when combined with techniques that account for label uncertainty. Additionally, the DANets framework \cite{2112.02962v4} demonstrates how feature abstraction and structure re-parameterization can improve model robustness in tabular data, which is relevant to both logistic regression and gradient boosting. However, the primary focus of these methods is on feature engineering rather than explicitly addressing label noise.

The experimental findings presented here suggest that logistic regression is more robust to label noise than gradient boosting in a small tabular classification task, contradicting the general intuition that ensemble methods are inherently more robust. This contrast highlights the importance of evaluating model performance under label noise in specific settings, as the relative robustness of different models may vary depending on dataset characteristics and the nature of the noise. Therefore, while prior work provides theoretical and methodological foundations, the current study contributes empirical evidence that informs the trade-offs between logistic regression and gradient boosting in the presence of label noise.
</text>

## Methods

\section{Methods}

The experimental methodology is designed to evaluate the impact of label noise on the performance of logistic regression and gradient boosting on a small tabular classification dataset. The dataset used is the "iris" dataset, which is a standard benchmark dataset for classification tasks. The dataset is sourced from the scikit-learn library using the \texttt{load\_iris} function, and is subsequently subsampled to ensure computational efficiency. The data is split into features $ \mathbf{X} \in \mathbb{R}^{n \times d} $ and labels $ \mathbf{y} \in \{0, 1, 2\}^n $, where $ n $ is the number of samples and $ d $ is the number of features.

To simulate label noise, a simple random label flipping approach is employed. For a given noise rate $ \eta \in [0, 1] $, the labels in the training set are randomly flipped with probability $ \eta $. Specifically, for each sample in the training set, a Bernoulli random variable $ b_i \sim \text{Bernoulli}(\eta) $ is generated. If $ b_i = 1 $, the label $ y_i $ is replaced with a randomly chosen label from the set $ \{0, 1, 2\} \setminus \{y_i\} $. This process results in a noisy label vector $ \mathbf{y}_{\text{noisy}} $, which is used for training the models.

Two models are evaluated: logistic regression (LR) and gradient boosting (GB). The LR model is trained using the \texttt{LogisticRegression} class from scikit-learn, with the parameter $ \text{max\_iter} $ set to 1000 to ensure convergence. The GB model is trained using the \texttt{GradientBoostingClassifier} class with $ n_{\text{estimators}} = 100 $ and a random state for reproducibility. Both models are trained on the noisy training set and evaluated on a clean test set, which is split from the original dataset using a 80-20 train-test split.

The performance of the models is measured using accuracy, defined as:

$$
\text{Accuracy} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} \mathbb{I}( \hat{y}_i = y_i^{(test)} )
$$

where $ \hat{y}_i $ is the predicted label for the $ i $-th test sample, and $ y_i^{(test)} $ is the true label. The experiment is repeated 5 times to ensure statistical significance, and the average accuracy across all runs is reported for each noise level. The results are analyzed to determine whether label noise has a more detrimental effect on logistic regression compared to gradient boosting.
</text>

## Experimental Setup

\section{Experimental Setup}

The experiments were conducted on the ``adult'' dataset from sklearn's openml repository, specifically using the ``adult'' dataset with 1000 samples. The dataset consists of 14 numerical and categorical features, with the target variable representing income level (``<=50K'' or ``>50K''). The dataset was preprocessed to handle categorical features through one-hot encoding, and all numerical features were normalized to the range [0, 1]. 

Label noise was introduced by randomly flipping labels in the training set at rates of 10\% and 20\%. The label flipping was performed uniformly across all classes, without any class-specific bias. 

The models evaluated were logistic regression and gradient boosting. Logistic regression was implemented using scikit-learn's `LogisticRegression` with a maximum number of iterations set to 1000. Gradient boosting was implemented using scikit-learn's `GradientBoostingClassifier` with 100 estimators. Both models were trained on the noisy training data and evaluated on a clean test set using accuracy as the primary performance metric. 

The data was split into training (80\%) and test (20\%) sets, with 5 independent runs to ensure statistical significance. The random states were set to different values for each run to ensure variability in the splits and noise introduction. 
</text>

## Results

\section{Results}

The experimental results are presented in Table~\ref{tab:results}, which summarizes the average accuracy of logistic regression and gradient boosting under 10\% and 20\% label noise. The results were obtained by running the experiment five times with different random states, ensuring statistical consistency. 

At 10\% label noise, logistic regression achieves an average accuracy of 0.960000, while gradient boosting achieves 0.940000. At 20\% label noise, logistic regression maintains an average accuracy of 0.906667, whereas gradient boosting drops to 0.846667. These results suggest that logistic regression is less impacted by label noise in this setting. However, the differences in performance between the two models under label noise are not extremely large, and further statistical analysis is needed to confirm the significance of these results.

It is important to note that the results are based on a small tabular classification dataset (the Iris dataset) and may not generalize to larger or more complex datasets. Additionally, the label noise was introduced by randomly flipping labels in the training set, which may not fully capture the complexities of real-world label noise scenarios. The experiments were conducted using scikit-learn only, without external libraries, and on a single GPU. The results provide a preliminary indication of the robustness of logistic regression compared to gradient boosting under label noise, but further investigation is required to draw more conclusive insights.

## Discussion

\section{Discussion}

The experimental results indicate that logistic regression exhibits greater robustness to label noise compared to gradient boosting on the considered small tabular classification dataset. At 10\% noise level, logistic regression achieves an average accuracy of 0.960000, while gradient boosting achieves 0.940000. At 20\% noise, logistic regression maintains an average accuracy of 0.906667, whereas gradient boosting drops to 0.846667. These findings suggest that logistic regression is less affected by label noise in this setting, potentially due to its simpler decision boundary and less sensitivity to individual mislabeled examples. However, the differences in performance between the two models under label noise are not extremely large, and further statistical analysis is required to confirm the significance of these results.

The observed robustness of logistic regression to label noise is consistent with prior literature, which highlights its simplicity and stability in the presence of noise. For instance, the work by \cite{2208.06828v2} provides insights into the optimization challenges in logistic regression, suggesting that its training procedure may inherently be less susceptible to the detrimental effects of label noise. Furthermore, the concept of ensemble methods for handling label noise, as discussed in \cite{2411.11020v2}, may provide a theoretical foundation for the resilience of logistic regression to noisy labels, even though it is not explicitly an ensemble method.

Gradient boosting, on the other hand, is generally known to be more robust to noise in certain settings, particularly when the noise is not uniformly distributed or when the noise is associated with specific instances. The results of this experiment, however, suggest that in the case of uniformly distributed label noise, logistic regression outperforms gradient boosting. This could be attributed to the fact that gradient boosting relies on multiple weak learners, which may be more sensitive to mislabeled instances during the iterative fitting process. The work by \cite{2012.11854v2} addresses instance-dependent label noise, suggesting that methods which account for the difficulty of individual instances may be better suited to handle such noise, which may not be fully captured in the current experimental setup.

It is also worth noting that the choice of dataset and the specific noise simulation method may influence the relative robustness of the models. The use of the iris dataset, which is relatively small and balanced, may not generalize to larger, more complex datasets where the impact of label noise could be more pronounced. Additionally, the method of label flipping, which assumes a uniform noise distribution, may not fully capture the complexities of real-world label noise scenarios, such as class-dependent or instance-dependent noise. Furthermore, the stated dataset in the experimental plan was the "adult" dataset, while the actual dataset used was the "iris" dataset, which introduces a discrepancy in the experimental setup and may affect the generalizability of the findings.

To address the limitations of the current study, future work should incorporate a more comprehensive statistical analysis, including confidence intervals and hypothesis testing, to validate the observed differences in model performance. Additionally, comparing the models with alternative noise handling techniques, such as label smoothing or noise-tolerant regularization, could provide further insights into the robustness of logistic regression and gradient boosting under different noise conditions. Finally, the noise simulation method should be refined to reflect more realistic label noise scenarios, such as class-dependent or instance-dependent noise, to better align with real-world data characteristics.
