# MCEClassifier

Code for paper `Ning, Zhihan, Zhixing Jiang, and David Zhang. "Exploiting Meta-Learned Confidences for Imbalanced Multilabel Learning." IEEE Transactions on Neural Networks and Learning Systems (2024).`

* Required Python 3 packages:
    1. `numpy==1.21.5`;
    2. `sklearn` (https://github.com/scikit-learn/scikit-learn);
    3. `scipy==1.7.3`.

* Optional Python 3 packages: 
    1. `imblearn` (https://github.com/scikit-learn-contrib/imbalanced-learn);
    2. `iterative-stratification` (https://github.com/trent-b/iterative-stratification).

* MCEClassifier is compatible with most sklearn APIs but is not strictly tested.

* Import: `from MCEClassifier import MCEClassifier`.

* Train: `fit(X, y)`, with target $\textbf{y}_i \in (0, 1)^l$ as the labels. 

* Predict: `predict(X)` (deterministic prediction), `predict_proba(X)` (probalistic prediction).

* Parameters: 
    1. `base_estimator`: Multi-label classifier object with `predict_proba()` function;
    2. `n_estimators`: The ensemble size $s$;
    3. `projection_ratio`: Controls the projection dimensionality, denoted as $\mu$;
    4. `projection_density`: Controls the average number of features to be projected, denoted as $\lambda$;
    5. `ratio_sampling`: The proportion of instances to be drawn as the training set in each iteration, denoted as $r$.
