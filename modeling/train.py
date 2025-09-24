"""
Run:
    python -m Happy_Customers.modeling.train
"""

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

from Happy_Customers.dataset import load_raw
from Happy_Customers.features import build_preprocessor  # we won't use it heavily, but it's available

# ---------------- Paths & constants ----------------
ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGS_DIR = REPORTS_DIR / "figures"
for d in (MODELS_DIR, REPORTS_DIR, FIGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# --------------- Helpers ----------------
def metrics_on_test(model, X_tr, y_tr, X_te, y_te, label):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    out = {
        "model": label,
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
    }
    # ROC-AUC if proba available
    try:
        y_proba = model.predict_proba(X_te)[:, 1]
        out["roc_auc"] = float(roc_auc_score(y_te, y_proba))
    except Exception:
        out["roc_auc"] = float("nan")
    return out, y_pred

def score_with_grid(estimator, grid, X_tr, y_tr, label, scoring="accuracy"):
    gs = GridSearchCV(estimator, grid, scoring=scoring, cv=CV, n_jobs=-1, refit=True, return_train_score=False)
    gs.fit(X_tr, y_tr)
    cv_tbl = pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False)
    cv_tbl.to_csv(REPORTS_DIR / f"gridsearch_{label}.csv", index=False)
    return gs.best_estimator_, float(gs.best_score_), gs.best_params_

def barplot_series(series, title, ylabel, out_png):
    plt.figure(figsize=(6,4))
    series.plot(kind="bar")
    plt.title(title); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def greedy_forward(X_tr, y_tr, X_te, y_te, base_estimator, features, scoring="accuracy"):
    selected, history = [], []
    remaining = list(features)
    best_score = -np.inf

    def eval_subset(cols):
        mdl = base_estimator
        mdl.fit(X_tr[cols], y_tr)
        y_pred = mdl.predict(X_te[cols])
        if scoring == "f1":
            return f1_score(y_te, y_pred, zero_division=0)
        return accuracy_score(y_te, y_pred)

    while remaining:
        best_feat, best_round = None, best_score
        for f in remaining:
            score = eval_subset(selected + [f])
            if score > best_round:
                best_round, best_feat = score, f
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_round
        history.append({"k_features": len(selected), "feature_added": best_feat, "score": float(best_round)})

    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        # plot curve
        plt.figure()
        plt.plot(hist_df["k_features"], hist_df["score"], marker="o")
        plt.xlabel("Number of features"); plt.ylabel(scoring.upper())
        plt.title(f"Greedy forward selection ({getattr(base_estimator, '__class__', type(base_estimator)).__name__})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIGS_DIR / "forward_selection_curve.png", bbox_inches="tight")
        plt.close()
    return selected, hist_df

# --------------- Main ----------------
def main():
    # Load
    df = load_raw()
    expected = [f"X{i}" for i in range(1, 6+1)]
    feats = [c for c in expected if c in df.columns]
    if "Y" not in df.columns:
        raise ValueError("Target column 'Y' not found.")
    if len(feats) == 0:
        raise ValueError("No X1..X6 features found.")
    X_all = df[feats].copy()
    y_all = df["Y"].astype(int).copy()

    # Fixed split to match notebook
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.25, random_state=RANDOM_STATE, stratify=y_all
    )

    # ----- Baselines (as in NB) -----
    models = {
        "LogReg(cw=None)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
        ]),
        "LogReg(cw=balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced"))
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }

    rows = []
    preds_store = {}

    for name, mdl in models.items():
        r, ypred = metrics_on_test(mdl, X_tr, y_tr, X_te, y_te, name)
        rows.append(r)
        preds_store[name] = ypred

    # ----- Feature importance (LR + RF) -----
    # LR coefficients (standardized space)
    lr_pipe = models["LogReg(cw=None)"]
    lr_pipe.fit(X_tr, y_tr)
    if hasattr(lr_pipe.named_steps["clf"], "coef_"):
        lr_coef = pd.Series(lr_pipe.named_steps["clf"].coef_.ravel(), index=feats).reindex(feats)
        lr_imp = lr_coef.abs().sort_values(ascending=False)
        lr_imp.to_csv(REPORTS_DIR / "importance_logreg.csv", header=["abs_coef"])
        barplot_series(lr_imp, "Logistic Regression | |Coefficient| by Feature", "abs(coef)", FIGS_DIR / "importance_logreg.png")

    # RF importances
    rf = models["RandomForest"]
    rf.fit(X_tr, y_tr)
    if hasattr(rf, "feature_importances_"):
        rf_imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
        rf_imp.to_csv(REPORTS_DIR / "importance_rf.csv", header=["feature_importance"])
        barplot_series(rf_imp, "RandomForest Feature Importances", "importance", FIGS_DIR / "importance_rf.png")

    # ----- Ranked subsets + grid search for SVC (RBF) & HGB -----
    # Use a ranking like your notebook (tweak if your importances differ)
    ranked = list(lr_imp.index) if 'lr_imp' in locals() else feats[:]  # fallback
    # ensure at least this order if present in data
    preferred_order = ["X1", "X5", "X2", "X3", "X6", "X4"]
    ranked = [f for f in preferred_order if f in ranked] + [f for f in ranked if f not in preferred_order]

    subset_candidates = [
        ranked[:2],
        ranked[:3],
        ranked[:4],
        ranked[:5],
        ranked[:6],
    ]
    res = []

    # small helper to compute a complete metrics dict on TEST
    def eval_test(estimator, label, Xtr, ytr, Xte, yte):
        out, _ = metrics_on_test(estimator, Xtr, ytr, Xte, yte, label)
        return out

    # Grids close to what you showed
    svc_base = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])
    svc_grid = {
        "clf__C": [5, 8, 12, 30],
        "clf__gamma": ["scale", 0.3, 0.1],
    }

    hgb_base = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    hgb_grid = {
        "learning_rate": [0.03, 0.1],
        "l2_regularization": [0.0, 0.1],
        # You can add "max_depth": [None, 2, 3] if needed
    }

    for subset in subset_candidates:
        # SVC on subset
        svc_best, svc_cv, svc_params = score_with_grid(svc_base, svc_grid, X_tr[subset], y_tr, label="svc")
        svc_row = eval_test(svc_best, f"SVC_RBF | {subset}", X_tr[subset], y_tr, X_te[subset], y_te)
        svc_row["cv_mean_accuracy"] = svc_cv
        svc_row["best_params"] = svc_params
        res.append(svc_row)

        # HGB on subset
        hgb_best, hgb_cv, hgb_params = score_with_grid(hgb_base, hgb_grid, X_tr[subset], y_tr, label="hgb")
        hgb_row = eval_test(hgb_best, f"HGB | {subset}", X_tr[subset], y_tr, X_te[subset], y_te)
        hgb_row["cv_mean_accuracy"] = hgb_cv
        hgb_row["best_params"] = hgb_params
        res.append(hgb_row)

    # Combine with baselines
    metrics_df = pd.DataFrame(rows + res).sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(REPORTS_DIR / "top_candidates.csv", index=False)

    # Best by TEST accuracy (to match NB printout)
    best_row = metrics_df.iloc[0].to_dict()
    best_label = best_row["model"]
    # Refit best on the appropriate subset/full features for saving
    if best_label.startswith("SVC_RBF"):
        subset = eval(best_label.split("|")[1].strip())  # turn "['X1','X5']" into list safely
        best_est, _, _ = score_with_grid(svc_base, svc_grid, X_tr[subset], y_tr, label="svc")
        Xtr_fit, Xte_fit = X_tr[subset], X_te[subset]
        used_features = subset
    elif best_label.startswith("HGB"):
        subset = eval(best_label.split("|")[1].strip())
        best_est, _, _ = score_with_grid(hgb_base, hgb_grid, X_tr[subset], y_tr, label="hgb")
        Xtr_fit, Xte_fit = X_tr[subset], X_te[subset]
        used_features = subset
    else:
        # one of the baselines
        best_est = {"LogReg(cw=None)": models["LogReg(cw=None)"],
                    "LogReg(cw=balanced)": models["LogReg(cw=balanced)"],
                    "RandomForest": models["RandomForest"]}[best_label]
        Xtr_fit, Xte_fit = X_tr, X_te
        used_features = feats

    # Final holdout report (and classification_report.txt)
    best_est.fit(Xtr_fit, y_tr)
    y_pred = best_est.predict(Xte_fit)
    try:
        y_proba = best_est.predict_proba(Xte_fit)[:, 1]
        auc = float(roc_auc_score(y_te, y_proba))
    except Exception:
        y_proba, auc = None, float("nan")

    final_metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": None if math.isnan(auc) else auc,
    }

    # Save classification report
    (REPORTS_DIR / "classification_report.txt").write_text(classification_report(y_te, y_pred, digits=6))

    # Permutation importance on holdout for the chosen features/model
    try:
        perm = permutation_importance(best_est, Xte_fit, y_te, n_repeats=20, random_state=RANDOM_STATE)
        perm_df = pd.DataFrame({
            "feature": used_features,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std
        }).sort_values("perm_importance_mean", ascending=False)
        perm_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
        barplot_series(perm_df.set_index("feature")["perm_importance_mean"],
                       f"Permutation Importance ({best_label})", "mean importance",
                       FIGS_DIR / "permutation_importance.png")
    except Exception as e:
        (REPORTS_DIR / "permutation_importance_error.txt").write_text(str(e))

    # Greedy forward selection (use LR for speed & stability)
    base_lr = Pipeline([("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))])
    selected, hist_df = greedy_forward(X_tr, y_tr, X_te, y_te, base_lr, feats, scoring="accuracy")
    if hist_df is not None and not hist_df.empty:
        hist_df.to_csv(REPORTS_DIR / "forward_selection_history.csv", index=False)

    # Persist best model
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_est, model_path)

    # Summary JSON
    summary = {
        "best_model_label": best_label,
        "used_features": used_features,
        "final_holdout_metrics": final_metrics,
        "top_candidates_path": str(REPORTS_DIR / "top_candidates.csv"),
        "gridsearch_svc_path": str(REPORTS_DIR / "gridsearch_svc.csv"),
        "gridsearch_hgb_path": str(REPORTS_DIR / "gridsearch_hgb.csv"),
        "classification_report": str(REPORTS_DIR / "classification_report.txt"),
        "model_path": str(model_path),
    }
    (REPORTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nTop candidates saved to reports/top_candidates.csv")
    print(f"Best: {best_label}  |  Holdout F1: {final_metrics['f1']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Summary: {REPORTS_DIR/'training_summary.json'}")
    print(f"Figures in: {FIGS_DIR}")

if __name__ == "__main__":
    main()
