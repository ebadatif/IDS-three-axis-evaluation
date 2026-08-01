"""
Classical baseline training (Random Forest and XGBoost) and the feature-
leakage audit that led to the locked 29-feature set.

Provides:
    train_baselines     - fit RF and XGBoost on UNSW same-dataset
    leakage_audit       - solo-F1 per feature + importance concentration
    ablation_experiment - the A/B/C feature-set comparison from Block 3.8
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb

from .config import FEATURES, LABEL, DEFAULT_SEED


def train_baselines(df, features=FEATURES, seed=DEFAULT_SEED):
    """Train Random Forest and XGBoost on a same-dataset 70/30 split.

    Returns (rf, xg, X_test, y_test) so downstream code can score them.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[features], df[LABEL],
        test_size=0.3, random_state=seed, stratify=df[LABEL],
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    xg = xgb.XGBClassifier(n_estimators=100, random_state=seed, n_jobs=-1,
                           eval_metric="logloss")
    xg.fit(X_tr, y_tr)

    return rf, xg, X_te, y_te


def leakage_audit(X_tr, X_te, y_tr, y_te, features=FEATURES, top_n=12):
    """Audit a feature set for leakage.

    Two diagnostics:
      1. Solo F1: fit a one-question decision stump on each feature alone.
         A single feature scoring > 0.90 solo F1 essentially IS the label
         in disguise - almost certainly leakage or an artifact.
      2. Full-model F1 + importance concentration: fit a Random Forest on
         the whole feature set and check whether one feature dominates.

    In the UNSW-NB15 dataset, MIN_TTL alone reaches 0.9945 solo F1 - it
    encodes the sender OS and hop count, i.e. the lab wiring, not attack
    behavior. Same-dataset F1 alone cannot distinguish this from real
    signal; only cross-dataset transfer can.

    Returns (solo_df, importance_df, full_model_f1).
    """
    solo = []
    for f in features:
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        stump.fit(X_tr[[f]], y_tr)
        solo.append((f, f1_score(y_te, stump.predict(X_te[[f]]))))
    solo_df = (pd.DataFrame(solo, columns=["feature", "solo_f1"])
               .sort_values("solo_f1", ascending=False)
               .reset_index(drop=True))

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr[features], y_tr)
    full_f1 = f1_score(y_te, rf.predict(X_te[features]))

    imp = (pd.DataFrame({"feature": features,
                         "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))

    return solo_df, imp, full_f1


def ablation_experiment(df_train, df_test, feature_sets, seed=DEFAULT_SEED):
    """Run the same-dataset vs cross-dataset comparison across feature sets.

    This is the experiment that resolves the artifact-vs-signal question:
    a feature is an "artifact" if its predictive power does not transfer to
    a different network. Cross-dataset transfer IS the operational test.

    Parameters
    ----------
    df_train, df_test : DataFrames from different networks (e.g. UNSW, CIC).
    feature_sets : dict mapping set-name -> list of feature names.

    Returns a DataFrame with SD_f1, CD_f1, CD_fnr for each feature set.
    """
    rows = []
    for name, feats in feature_sets.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            df_train[feats], df_train[LABEL],
            test_size=0.3, random_state=seed, stratify=df_train[LABEL],
        )

        model = xgb.XGBClassifier(n_estimators=100, random_state=seed,
                                  n_jobs=-1, eval_metric="logloss")
        model.fit(X_tr, y_tr)

        sd_f1 = f1_score(y_te, model.predict(X_te))

        cd_pred = model.predict(df_test[feats])
        cd_f1 = f1_score(df_test[LABEL], cd_pred)
        tn, fp, fn, tp = confusion_matrix(df_test[LABEL], cd_pred).ravel()
        cd_fnr = fn / (fn + tp)

        rows.append({
            "set": name, "n_feat": len(feats),
            "SD_f1": round(sd_f1, 4),
            "CD_f1": round(cd_f1, 4),
            "CD_fnr": round(cd_fnr, 4),
        })
    return pd.DataFrame(rows)
