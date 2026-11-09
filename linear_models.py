"""Лабораторная №3: собственные линейная и логистическая регрессии (batch GD)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42
_EPS = 1e-12


# ----------------------------------------------------------------- helpers
def _to_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _to_1d(y) -> np.ndarray:
    return np.asarray(y, dtype=float).ravel()


def _fit_scaler(X: np.ndarray, standardize: bool):
    """Среднее и std по train. Константные признаки не делятся на 0."""
    if not standardize:
        return np.zeros(X.shape[1]), np.ones(X.shape[1])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # устойчивая форма: не даёт overflow в exp при больших |z|
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _unscale(w_s: np.ndarray, b_s: float, mean: np.ndarray, std: np.ndarray):
    """Перевод параметров из стандартизованного пространства в исходное."""
    w = w_s / std
    b = float(b_s - np.sum(w_s * mean / std))
    return w, b


def fit_linear_gd(X, y, learning_rate, epochs, standardize=True):
    """Пакетный GD для MSE. Возвращает (w, b, история MSE) в исходных единицах."""
    X, y = _to_2d(X), _to_1d(y)
    mean, std = _fit_scaler(X, standardize)
    Z = (X - mean) / std
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    history: list[float] = []
    with np.errstate(all="ignore"):
        for _ in range(int(epochs)):
            err = Z @ w + b - y
            history.append(float(np.mean(err ** 2)))
            w = w - learning_rate * (2.0 / n) * (Z.T @ err)
            b = b - learning_rate * (2.0 / n) * err.sum()
    w, b = _unscale(w, b, mean, std)
    return w, b, history


def fit_logistic_gd(X, y, learning_rate, epochs, standardize=True):
    """Пакетный GD для бинарной кросс-энтропии. Возвращает (w, b, история BCE)."""
    X, y = _to_2d(X), _to_1d(y)
    mean, std = _fit_scaler(X, standardize)
    Z = (X - mean) / std
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    history: list[float] = []
    for _ in range(int(epochs)):
        p = _sigmoid(Z @ w + b)
        pc = np.clip(p, _EPS, 1.0 - _EPS)
        history.append(float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))))
        err = p - y
        w = w - learning_rate * (Z.T @ err) / n
        b = b - learning_rate * err.sum() / n
    w, b = _unscale(w, b, mean, std)
    return w, b, history


def classification_metrics(y_true, proba, pred=None) -> ClassificationMetrics:
    y_true = _to_1d(y_true).astype(int)
    proba = _to_1d(proba)
    pred = (proba >= 0.5).astype(int) if pred is None else _to_1d(pred).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, proba)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
    )


# ------------------------------------------------------------- public API
def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    w, b, history = fit_linear_gd(
        data.train_features, data.train_target, data.learning_rate, data.epochs
    )
    predictions = _to_2d(data.test_features) @ w + b
    return RegressionResult(weights=w, bias=b, loss_history=history, predictions=predictions)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    w, b, history = fit_logistic_gd(
        data.train_features, data.train_target, data.learning_rate, data.epochs
    )
    proba = _sigmoid(_to_2d(data.test_features) @ w + b)
    return ClassificationResult(
        weights=w,
        bias=b,
        loss_history=history,
        probabilities=proba,
        predictions=(proba >= 0.5).astype(int),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    Поле logistic_regression здесь — sklearn LogisticRegression (как в контракте
    и в ключах experiment_results.json); собственная модель сравнивается с ним
    отдельно в run_experiments.py (ключ own_logistic).
    Масштабирование для LR и KNN обучается только на train.
    """
    Xtr, ytr = _to_2d(data.train_features), _to_1d(data.train_target).astype(int)
    Xte, yte = _to_2d(data.test_features), _to_1d(data.test_target).astype(int)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        metrics[name] = classification_metrics(
            yte, model.predict_proba(Xte)[:, 1], model.predict(Xte)
        )
    return BaselineComparisonResult(**metrics)
