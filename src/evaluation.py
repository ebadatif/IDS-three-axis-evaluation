"""
Fair evaluation harness. Every model in this study is scored by the SAME
function so that classical and LLM results are comparable by construction.

Provides:
    evaluate            - one metric definition for every model, everywhere
    run_experiment      - one model, one condition, one seed
    run_full_benchmark  - one model, all conditions, all seeds
    summarize           - collapse seeds into mean +/- std for reporting
"""

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, balanced_accuracy_score,
)

from .config import FEATURES, LABEL, SEEDS


def evaluate(y_true, y_pred, model_name="", condition="", seed=None):
    """Return one row of metrics for a single (model, condition, seed) result.

    Reports F1, precision, recall, balanced accuracy, FPR, FNR, and the raw
    confusion-matrix cells (needed later for figures without re-running).

    FNR (fraction of attacks missed) is reported alongside F1 because it
    matters more than F1 in security: a missed attack costs more than a
    false alarm. Mehavilla et al. found similar F1 but different FNR.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": model_name,
        "condition": condition,
        "seed": seed,
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "bal_acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "fpr": round(fp / (fp + tn), 4),
        "fnr": round(fn / (fn + tp), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_experiment(model_class, model_params, df_train, df_test,
                   condition, seed, features=FEATURES):
    """Train a fresh model instance and score it through the harness.

    For same-dataset conditions ("SD_*"), splits df_train 70/30 internally.
    For cross-dataset conditions ("CD_*"), trains on all of df_train and
    tests on all of df_test.

    Returns
    -------
    (result_dict, trained_model)
    """
    if condition.startswith("SD"):
        X_tr, X_te, y_tr, y_te = train_test_split(
            df_train[features], df_train[LABEL],
            test_size=0.3, random_state=seed, stratify=df_train[LABEL],
        )
    else:
        X_tr, y_tr = df_train[features], df_train[LABEL]
        X_te, y_te = df_test[features], df_test[LABEL]

    model = model_class(**model_params, random_state=seed)
    model.fit(X_tr, y_tr)

    t0 = time.time()
    preds = model.predict(X_te)
    elapsed = time.time() - t0

    result = evaluate(y_te, preds,
                      model_name=model_class.__name__,
                      condition=condition, seed=seed)
    result["n_test"] = len(X_te)
    result["infer_sec"] = round(elapsed, 4)
    result["flows_sec"] = int(len(X_te) / max(elapsed, 1e-9))
    return result, model


def run_full_benchmark(model_class, model_params, df_unsw, df_cic,
                       seeds=SEEDS):
    """Run one model across all four conditions and all seeds.

    Conditions:
        SD_unsw       - train and test on UNSW (70/30 split)
        SD_cic        - train and test on CIC (70/30 split)
        CD_unsw2cic   - train on all of UNSW, test on all of CIC
        CD_cic2unsw   - train on all of CIC, test on all of UNSW

    Returns a DataFrame with one row per (condition, seed) pair.
    """
    all_results = []
    for seed in seeds:
        for train_df, test_df, cond in [
            (df_unsw, df_unsw, "SD_unsw"),
            (df_cic, df_cic, "SD_cic"),
            (df_unsw, df_cic, "CD_unsw2cic"),
            (df_cic, df_unsw, "CD_cic2unsw"),
        ]:
            r, _ = run_experiment(model_class, model_params,
                                  train_df, test_df, cond, seed)
            all_results.append(r)
    return pd.DataFrame(all_results)


def summarize(results_df, metrics=("f1", "precision", "recall",
                                    "bal_acc", "fpr", "fnr", "flows_sec")):
    """Compute mean and std across seeds, grouped by (model, condition).

    Notes
    -----
    XGBoost cross-dataset std may be 0.0 because CD conditions use
    all-train / all-test with no split randomness, and XGBoost with fixed
    hyperparameters is deterministic enough that seed changes don't alter
    predictions on 100k rows. Report this in the paper as a methodological
    note, not a bug.
    """
    summary = (results_df.groupby(["model", "condition"])[list(metrics)]
               .agg(["mean", "std"])
               .round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary
