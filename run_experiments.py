"""Воспроизводимый запуск экспериментов Лабораторной №3.

python run_experiments.py  ->  experiment_results.json, loss_curves.png, roc_curve.png
"""
from __future__ import annotations

import json
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    RANDOM_STATE, _sigmoid, classification_metrics, compare_classical_models,
    fit_linear_gd, fit_logistic_gd, train_linear_regression, train_logistic_classifier,
)

TEST_SIZE = 0.2
out: dict = {}


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


# ------------------------------------------------------------ regression
st = pd.read_csv("Student_Performance.csv")
st["Extracurricular Activities"] = st["Extracurricular Activities"].map({"No": 0, "Yes": 1})
assert st.isna().sum().sum() == 0
X = st.drop(columns="Performance Index").to_numpy(float)
y = st["Performance Index"].to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

reg = train_linear_regression(RegressionInput(Xtr, ytr, Xte, yte))
reg_mse = mse(yte, reg.predictions)
sk = LinearRegression().fit(Xtr, ytr)  # только как эталон для проверки
reg_ref = {
    "own_gd_test_mse": reg_mse,
    "sklearn_linear_regression_test_mse": float(mean_squared_error(yte, sk.predict(Xte))),
    "mean_predictor_test_mse": mse(yte, np.full_like(yte, ytr.mean())),
    "max_abs_weight_diff_vs_sklearn": float(np.max(np.abs(reg.weights - sk.coef_))),
    "bias_diff_vs_sklearn": float(abs(reg.bias - sk.intercept_)),
    "weights": dict(zip(st.columns[:-1], map(float, reg.weights))),
    "bias": float(reg.bias),
    "train_mse_epoch_1": reg.loss_history[0],
    "train_mse_epoch_10": reg.loss_history[9],
    "train_mse_epoch_100": reg.loss_history[99],
    "train_mse_epoch_800": reg.loss_history[-1],
}
# гипотезы
reg_exp = {}
w, b, h = fit_linear_gd(Xtr, ytr, 0.05, 800, standardize=False)
reg_exp["no_scaling_lr0.05"] = {"final_train_mse": h[-1], "test_mse": mse(yte, Xte @ w + b)}
w, b, h = fit_linear_gd(Xtr, ytr, 1e-4, 800, standardize=False)
reg_exp["no_scaling_lr1e-4"] = {"final_train_mse": h[-1], "test_mse": mse(yte, Xte @ w + b)}
for lr in (0.001, 0.01, 0.05, 0.2, 0.5):
    w, b, h = fit_linear_gd(Xtr, ytr, lr, 800)
    reg_exp[f"scaled_lr{lr}"] = {"final_train_mse": h[-1], "test_mse": mse(yte, Xte @ w + b)}
for ep in (10, 50, 200, 800):
    w, b, h = fit_linear_gd(Xtr, ytr, 0.05, ep)
    reg_exp[f"scaled_lr0.05_epochs{ep}"] = {"test_mse": mse(yte, Xte @ w + b)}
reg_ref["experiments"] = reg_exp

# ---------------------------------------------------------- classification
g = pd.read_csv("german.csv", sep=";")  # файл разделён «;» — по умолчанию читается 1 столбец
yc = g["Creditability"].to_numpy(int)
Xc = g.drop(columns="Creditability").to_numpy(float)
Xctr, Xcte, yctr, ycte = train_test_split(
    Xc, yc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yc
)
cdata = ClassificationInput(Xctr, yctr, Xcte, ycte)
clf = train_logistic_classifier(cdata)
own = classification_metrics(ycte, clf.probabilities, clf.predictions)
base = compare_classical_models(cdata)

sk_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xctr, yctr)
cls_ref = {
    "n_train": int(len(yctr)), "n_test": int(len(ycte)),
    "test_share_of_class_1": float(ycte.mean()),
    "majority_class_accuracy": float(max(ycte.mean(), 1 - ycte.mean())),
    "max_abs_proba_diff_own_vs_sklearn_lr": float(
        np.max(np.abs(clf.probabilities - sk_lr.predict_proba(Xcte)[:, 1]))),
    "bce_epoch_1": clf.loss_history[0], "bce_epoch_100": clf.loss_history[99],
    "bce_epoch_1000": clf.loss_history[-1],
}
cls_exp = {}
for lr in (0.001, 0.01, 0.1, 0.5):
    w, b, h = fit_logistic_gd(Xctr, yctr, lr, 1000)
    m = classification_metrics(ycte, _sigmoid(Xcte @ w + b))
    cls_exp[f"scaled_lr{lr}"] = {"final_train_bce": h[-1], **asdict(m)}
for ep in (50, 200, 1000, 5000):
    w, b, h = fit_logistic_gd(Xctr, yctr, 0.1, ep)
    m = classification_metrics(ycte, _sigmoid(Xcte @ w + b))
    cls_exp[f"scaled_lr0.1_epochs{ep}"] = {"final_train_bce": h[-1], **asdict(m)}
w, b, h = fit_logistic_gd(Xctr, yctr, 0.1, 1000, standardize=False)
m = classification_metrics(ycte, _sigmoid(Xcte @ w + b))
cls_exp["no_scaling_lr0.1"] = {"final_train_bce": h[-1], **asdict(m)}
w, b, h = fit_logistic_gd(Xctr, yctr, 1e-5, 1000, standardize=False)
m = classification_metrics(ycte, _sigmoid(Xcte @ w + b))
cls_exp["no_scaling_lr1e-5"] = {"final_train_bce": h[-1], **asdict(m)}
# порог: 0.5 против 0.7 (дефолт-класс 0 «плохой заёмщик» дороже ошибки)
for thr in (0.5, 0.6, 0.7):
    m = classification_metrics(ycte, clf.probabilities, (clf.probabilities >= thr).astype(int))
    cls_exp[f"threshold_{thr}"] = asdict(m)
cls_ref["experiments"] = cls_exp

# устойчивость к разбиению: 10 seed
seeds = range(10)
acc = {k: [] for k in ("own_logistic", "logistic_regression", "decision_tree", "knn")}
auc = {k: [] for k in acc}
for s in seeds:
    a, bb, c, d = train_test_split(Xc, yc, test_size=TEST_SIZE, random_state=s, stratify=yc)
    di = ClassificationInput(a, c, bb, d)
    r = train_logistic_classifier(di)
    mm = {"own_logistic": classification_metrics(d, r.probabilities, r.predictions)}
    cb = compare_classical_models(di)
    mm.update(logistic_regression=cb.logistic_regression, decision_tree=cb.decision_tree, knn=cb.knn)
    for k, v in mm.items():
        acc[k].append(v.accuracy); auc[k].append(v.roc_auc)
cls_ref["ten_seeds"] = {
    k: {"roc_auc_mean": float(np.mean(auc[k])), "roc_auc_std": float(np.std(auc[k])),
        "accuracy_mean": float(np.mean(acc[k])), "accuracy_std": float(np.std(acc[k]))}
    for k in acc
}

# ---------------------------------------------------------------- output
result = {
    "linear_regression": {"test_mse": reg_mse},
    "classification": {
        "own_logistic": asdict(own),
        "logistic_regression": asdict(base.logistic_regression),
        "decision_tree": asdict(base.decision_tree),
        "knn": asdict(base.knn),
    },
}
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
def clean(o):
    """NaN/inf (расходимость GD) -> null, чтобы JSON был валидным."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


with open("experiment_details.json", "w", encoding="utf-8") as f:
    json.dump(clean({"regression": reg_ref, "classification": cls_ref}), f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(reg.loss_history); ax[0].set_yscale("log"); ax[0].set_title("Linear: train MSE")
ax[0].set_xlabel("epoch")
ax[1].plot(clf.loss_history); ax[1].set_title("Logistic: train BCE"); ax[1].set_xlabel("epoch")
plt.tight_layout(); plt.savefig("loss_curves.png", dpi=120)
plt.figure(figsize=(5, 4.5))
for name, p in [("own", clf.probabilities), ("sklearn LR", sk_lr.predict_proba(Xcte)[:, 1])]:
    fpr, tpr, _ = roc_curve(ycte, p); plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC, german.csv (test)"); plt.tight_layout(); plt.savefig("roc_curve.png", dpi=120)
print(json.dumps(result, indent=2, ensure_ascii=False))

