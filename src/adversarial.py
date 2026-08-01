"""
Adversarial evasion attack in feature space with domain constraints.

Strategy: "drift toward benign" - linearly interpolate attack flows toward
the benign centroid at a range of perturbation strengths epsilon in [0, 1].

At epsilon = 0 the flow is unmodified. At epsilon = 1 it equals the benign
centroid. In between, the attack is partially disguised as normal traffic.

The benign centroid is computed from the TRAINING split only. This mirrors
a realistic threat model: an attacker can observe or estimate normal traffic
patterns but cannot see the specific test flows the defender will encounter.

Feature-space attacks are the standard convention in the NIDS adversarial
literature (Pierazzi et al., Apruzzese et al.). We do not claim every
perturbed vector corresponds to an executable packet sequence; this is a
stated limitation. What we DO claim is that the same attack is applied
identically to both models, so the comparison between them is fair.
"""

import numpy as np
import pandas as pd

from .config import FEATURES, LABEL, ADV_FIXED_FEATURES


# Features that carry a physical constraint on their values.
# Auto-detected by name so this list stays maintainable.
def _non_negative_features(perturbable):
    keys = ("BYTES", "PKTS", "DURATION", "THROUGHPUT",
            "NUM_PKTS", "TTL", "WIN", "FLOW_DURATION", "RETRANSMITTED")
    return [f for f in perturbable if any(k in f for k in keys)]


def _integer_features(perturbable):
    keys = ("PKTS", "FLAGS", "NUM_PKTS")
    return [f for f in perturbable if any(k in f for k in keys)]


def apply_constraints(df, perturbable):
    """Enforce physical constraints on perturbed flows.

    Three constraints:
      1. Non-negative: counts and measurements can't be < 0.
      2. Integer: packet counts and flag encodings can't be fractional.
      3. RETRANSMITTED_* <= corresponding total: you can't retransmit
         more bytes/packets than you sent in the first place.
    """
    df = df.copy()
    non_neg = _non_negative_features(perturbable)
    ints = _integer_features(perturbable)

    for f in non_neg:
        if f in df.columns:
            df[f] = df[f].clip(lower=0)
    for f in ints:
        if f in df.columns:
            df[f] = df[f].round()

    # Retransmitted cannot exceed the corresponding total.
    for retrans, total in [
        ("RETRANSMITTED_IN_BYTES",  "IN_BYTES"),
        ("RETRANSMITTED_OUT_BYTES", "OUT_BYTES"),
        ("RETRANSMITTED_IN_PKTS",   "IN_PKTS"),
        ("RETRANSMITTED_OUT_PKTS",  "OUT_PKTS"),
    ]:
        if retrans in df.columns and total in df.columns:
            df[retrans] = df[[retrans, total]].min(axis=1)
    return df


def get_benign_centroid(df_train, perturbable):
    """Mean value of each perturbable feature over benign training flows."""
    return df_train[df_train[LABEL] == 0][perturbable].mean()


def perturb_toward_benign(df_attacks, centroid, epsilon, perturbable):
    """Linear interpolation between attack values and the benign centroid.

    new = (1 - epsilon) * original + epsilon * centroid

    epsilon = 0 -> unchanged; epsilon = 1 -> equals centroid.

    After interpolation, physical constraints are enforced and the result is
    cast to float32 for identical dtype presentation.
    """
    perturbed = df_attacks.copy()
    for f in perturbable:
        perturbed[f] = (1 - epsilon) * perturbed[f] + epsilon * centroid[f]
    perturbed = apply_constraints(perturbed, perturbable)
    perturbed[FEATURES] = perturbed[FEATURES].astype("float32")
    return perturbed


def build_evasion_suite(df_train, df_test_attacks, epsilons,
                        fixed_features=ADV_FIXED_FEATURES):
    """Generate the full set of perturbed attack samples across all epsilons.

    Parameters
    ----------
    df_train : DataFrame
        Training data used to derive the benign centroid (no test leakage).
    df_test_attacks : DataFrame
        Attack-only rows (LABEL == 1) from the held-out test set.
    epsilons : list of float in [0, 1]
    fixed_features : list of str
        Categorical / bounded features that must NOT be perturbed.

    Returns
    -------
    (adv_datasets, perturbable) where adv_datasets is dict[eps] -> DataFrame.
    """
    perturbable = [f for f in FEATURES if f not in fixed_features]
    centroid = get_benign_centroid(df_train, perturbable)
    adv_datasets = {
        eps: perturb_toward_benign(df_test_attacks, centroid, eps, perturbable)
        for eps in epsilons
    }
    return adv_datasets, perturbable


def score_evasion_curve(predict_fn, adv_datasets, model_name):
    """Run a predict_fn over each epsilon dataset and record detection rate.

    Every dataset in adv_datasets contains only attacks (true label = 1).
    Detection rate = fraction still classified as attack after perturbation.
    Lower detection rate at higher epsilon means the attack succeeded.

    predict_fn : callable(df) -> np.ndarray of 0/1 predictions.
    """
    rows = []
    for eps, df_adv in adv_datasets.items():
        preds = predict_fn(df_adv)
        detected = int(preds.sum())
        evaded = len(preds) - detected
        det_rate = detected / len(preds)
        rows.append({
            "model": model_name,
            "epsilon": eps,
            "detected": detected,
            "evaded": evaded,
            "detection_rate": round(det_rate, 4),
        })
    return pd.DataFrame(rows)
