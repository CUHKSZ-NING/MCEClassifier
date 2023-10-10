from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier as DT
from copy import deepcopy
import random
import numpy as np
from imblearn.metrics import geometric_mean_score
from scipy.stats import spearmanr


class DT_new(BaseEstimator):
    def __init__(self):
        self.model = DT()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        y_pred_proba = self.model.predict_proba(X)
        label_pred_proba = np.zeros((len(X), len(y_pred_proba)))
        for i in range(0, len(y_pred_proba)):
            try:
                label_pred_proba[..., i] = y_pred_proba[i][..., 1]
            except IndexError:
                label_pred_proba[..., i] = y_pred_proba[i][..., 0]
        return label_pred_proba


class PLEClassifier(BaseEstimator):
    def __init__(self, base_estimator=DT_new(), n_estimators=100, meta=True, projection_ratio=1.0, projection_density=3,
                 ratio_sampling=0.8, weights=None, re_balancing=True, entries='auto', feature_selection=False):
        if weights is None:
            weights = [0, 0, 0]
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.meta = meta
        self.projection_ratio = projection_ratio
        self.projection_density = int(projection_density + 0.5)
        self.ratio_sampling = ratio_sampling
        self.harmonizing_weight = weights[0]
        self.balancing_weight = weights[1]
        self.occurrence_weight = weights[2]
        self.re_balancing = re_balancing
        self.entries = entries
        self.feature_selection = feature_selection

        self.ensemble = {}
        self.projection_matrices = {}

        self.n_labels = 0
        self.n_features = 0
        self.n_features_original = 0
        self.n_instances = 0
        self.hardness_X = None
        self.scarcity_X = None
        self.scarcity_y = None
        self.oob_count_X = None
        self.temp = []
        self.threshold_target = None
        self.filters = {}
        self.oob_results = None

    def projection_matrix(self, iteration=-1):
        projection_dim = max(int(self.n_features * self.projection_ratio + 0.5), 1)
        projection = np.zeros((self.n_features, projection_dim))
        for i in range(0, projection_dim):
            if self.entries == 'auto' and self.meta is True:
                indices = []
                weight = np.zeros(self.n_features)
                for j in range(0, len(weight)):
                    if j < self.n_features_original:
                        weight[j] = 1
                    else:
                        weight[j] = iteration / self.n_estimators
                weight = weight * self.projection_density / sum(weight)
                for j in range(0, len(weight)):
                    if random.random() <= weight[j]:
                        indices.append(j)
                if len(indices) == 0:
                    indices = random.sample(range(0, self.n_features_original), 1)
            else:
                indices = random.sample(range(0, self.n_features), self.projection_density)
            for j in indices:
                projection[j, i] = random.uniform(-2, 2)
        return projection

    def fit(self, X, y):
        self.n_labels = len(y[0])
        if self.meta:
            self.n_features = len(X[0]) + self.n_labels
            self.n_features_original = len(X[0])
        else:
            self.n_features = len(X[0])
        self.n_instances = len(X)
        self.oob_results = np.zeros((self.n_instances, self.n_labels)) + 0.5

        for i in range(0, self.n_estimators):
            self.ensemble[i] = deepcopy(self.base_estimator)

        self.oob_count_X = np.zeros(self.n_instances)

        if self.harmonizing_weight > 0:
            self.hardness_X = np.zeros(self.n_instances) + 0.5

        if self.balancing_weight > 0:
            self.scarcity_X = np.zeros(self.n_instances)
            self.scarcity_y = np.zeros(self.n_labels)
            for i in range(0, self.n_labels):
                self.scarcity_y[i] = np.sum(y[..., i]) / self.n_instances
            self.scarcity_X = np.dot(y, 1 - self.scarcity_y.T) + np.dot((-1) * (y - 1), self.scarcity_y.T)

        for i in range(0, self.n_estimators):
            pseudo_labels = np.copy(self.oob_results)
            for j in range(0, self.n_labels):
                pseudo_labels[..., j] /= (self.oob_count_X + 1)

            if self.harmonizing_weight > 0:
                self.hardness_X = np.sum(np.abs(pseudo_labels - y), axis=1)

            X_RF = np.concatenate((X, pseudo_labels), axis=1)
            indices_train, indices_val = self.resampling()

            if self.projection_ratio > 0 and self.projection_density > 0:
                self.projection_matrices[i] = self.projection_matrix(i + 1)
                X_new = np.dot(X_RF[indices_train][..., 0: self.n_features], self.projection_matrices[i])
            else:
                X_new = X_RF[indices_train][..., 0: self.n_features]

            if self.feature_selection:
                self.filters[i] = self.filter_FS(X_new, y[indices_train])
                self.ensemble[i].fit(X_new[..., self.filters[i]], y[indices_train])
            else:
                self.ensemble[i].fit(X_new, y[indices_train])

            if len(self.projection_matrices) > 0:
                if self.feature_selection:
                    y_pred_proba = self.ensemble[i].predict_proba(
                        np.dot(X_RF[indices_val][..., 0: self.n_features],
                               self.projection_matrices[i])[..., self.filters[i]])
                else:
                    y_pred_proba = self.ensemble[i].predict_proba(
                        np.dot(X_RF[indices_val][..., 0: self.n_features], self.projection_matrices[i]))
            else:
                if self.feature_selection:
                    y_pred_proba = self.ensemble[i].predict_proba(
                        X_RF[indices_val][..., 0: self.n_features][..., self.filters[i]])
                else:
                    y_pred_proba = self.ensemble[i].predict_proba(X_RF[indices_val][..., 0: self.n_features])

            self.oob_results[indices_val] += y_pred_proba
            self.oob_count_X[indices_val] += 1

        if self.re_balancing:
            pseudo_labels = np.copy(self.oob_results)
            for j in range(0, self.n_labels):
                pseudo_labels[..., j] /= (self.oob_count_X + 1)
            self.distribution_alignment(y, pseudo_labels)

    def resampling(self):
        weight_X = np.zeros(self.n_instances)

        if self.occurrence_weight > 0:
            if np.max(self.oob_count_X) != np.min(self.oob_count_X):
                weight_o = self.oob_count_X - np.min(self.oob_count_X)
                weight_o /= np.max(weight_o)
                weight_X += weight_o * self.occurrence_weight

        if self.harmonizing_weight > 0:
            if np.max(self.hardness_X) != np.min(self.hardness_X):
                weight_h = self.hardness_X - np.min(self.hardness_X)
                weight_h /= np.max(weight_h)
                weight_X += weight_h * self.harmonizing_weight

        if self.balancing_weight > 0:
            if np.max(self.scarcity_X) != np.min(self.scarcity_X):
                weight_b = self.scarcity_X - np.min(self.scarcity_X)
                weight_b /= np.max(weight_b)
                weight_X += weight_b * self.balancing_weight

        for i in range(0, self.n_instances):
            weight_X[i] += random.random()

        threshold = sorted(weight_X)[int(self.n_instances * (1 - self.ratio_sampling) + 0.5)]
        indices_train = np.where(weight_X >= threshold)[0]
        indices_val = list(set(range(0, self.n_instances)) - set(indices_train))

        self.temp.append(len(indices_train) / np.sum(len(indices_train) + len(indices_val)))

        return indices_train, indices_val

    def filter_FS(self, X, y):
        importance = np.zeros(len(X[0]))
        for i in range(0, len(X[0])):
            for j in range(0, self.n_labels):
                correlation = np.abs(spearmanr(X[..., i], y[..., j])[0])
                if correlation == correlation:
                    importance[i] += correlation

        return np.where(importance >= sorted(importance)[int(len(importance) * 1 / 3 - 0.5)])[0]

    def distribution_alignment(self, y_true, y_scores):
        self.threshold_target = np.zeros(self.n_labels) + 0.5
        y_scores[np.where(self.oob_count_X == 0)] += 0.5

        for i in range(0, self.n_labels):
            score = 0
            for j in range(0, 100):
                y_temp = np.zeros(len(y_true), dtype=int)
                y_temp[np.where(y_scores[..., i] > 0.01 * j)] = 1
                score_temp = geometric_mean_score(y_true[..., i], y_temp)
                # score_temp += f1_score(y_true[..., i], y_temp)
                if score_temp >= score:
                    self.threshold_target[i] = 0.01 * j
                    score = score_temp

    def predict(self, X):
        pass

    def predict_proba(self, X):
        n_instances = len(X)
        if self.meta:
            X_RF = np.concatenate((X, np.zeros((n_instances, self.n_labels)) + 0.5), axis=1)
        else:
            X_RF = np.copy(X)
        label_pred_proba = np.zeros((n_instances, self.n_labels))

        for i in range(0, self.n_estimators):
            if len(self.projection_matrices) > 0:
                if self.feature_selection:
                    y_pred_proba = self.ensemble[i].predict_proba(
                        np.dot(X_RF, self.projection_matrices[i])[..., self.filters[i]])
                else:
                    y_pred_proba = self.ensemble[i].predict_proba(np.dot(X_RF, self.projection_matrices[i]))
            else:
                if self.feature_selection:
                    y_pred_proba = self.ensemble[i].predict_proba(X_RF[..., self.filters[i]])
                else:
                    y_pred_proba = self.ensemble[i].predict_proba(X_RF)
            if self.meta:
                temp = X_RF[..., (-1) * self.n_labels:]
                X_RF[..., (-1) * self.n_labels:] = (y_pred_proba + (i + 1) * temp) / (i + 2)
            label_pred_proba += y_pred_proba / len(self.ensemble)

        if self.re_balancing:
            for i in range(0, self.n_labels):
                weight_positive = label_pred_proba[..., i] / self.threshold_target[i]
                weight_negative = (1 - label_pred_proba[..., i]) / (1 - self.threshold_target[i])
                label_pred_proba[..., i] = weight_positive / (weight_positive + weight_negative)

        return label_pred_proba
