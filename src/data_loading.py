"""
Data loading and cleaning for the NetFlow v2 datasets.

Provides:
    load_balanced   - stream a parquet file and return a class-balanced sample
    apply_caps      - clip extreme/non-finite values using training-derived caps
    load_and_clean  - convenience wrapper that loads both datasets and cleans them

The NF-CSE-CIC-IDS2018-v2 file is 17M+ rows. `load_balanced` streams it in
chunks so the full file never enters RAM at once.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import FEATURES, LABEL, N_PER_CLASS, DEFAULT_SEED


def load_balanced(path, n_per_class=N_PER_CLASS, seed=DEFAULT_SEED,
                  batch_size=1_000_000):
    """Stream a parquet file and return a class-balanced sample.

    Keeps every attack row (they are the minority) and randomly subsamples
    benign rows during streaming, so peak memory is bounded by batch_size.

    Parameters
    ----------
    path : str
        Path to a NetFlow v2 parquet file.
    n_per_class : int
        Target number of samples per class (benign and attack). If either
        class has fewer available, both classes are shrunk to match.
    seed : int
        Random seed for reproducible sampling.
    batch_size : int
        Rows per streaming batch. Larger = fewer batches but more RAM.

    Returns
    -------
    pandas.DataFrame with 2 * n_per_class rows, shuffled, float32 features.
    """
    cols = FEATURES + [LABEL]
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows

    # Oversample benign 8x so we always have enough to draw the final sample.
    keep_frac = min(1.0, (n_per_class * 8) / total_rows)

    ben_parts, att_parts = [], []
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        b = batch.to_pandas()
        att_parts.append(b[b[LABEL] == 1])
        ben = b[b[LABEL] == 0]
        if keep_frac < 1.0:
            ben = ben.sample(frac=keep_frac, random_state=seed)
        ben_parts.append(ben)

    benign = pd.concat(ben_parts, ignore_index=True)
    attack = pd.concat(att_parts, ignore_index=True)

    # If a class is short, both classes shrink together to preserve balance.
    n = min(n_per_class, len(benign), len(attack))
    if n < n_per_class:
        print(f"  Warning: only {n:,} per class available (wanted {n_per_class:,})")

    benign = benign.sample(n=n, random_state=seed)
    attack = attack.sample(n=n, random_state=seed)

    df = pd.concat([benign, attack], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df


def compute_caps(df, features=FEATURES, percentile=99.9):
    """Compute per-feature clipping caps from training data.

    Some CIC features (SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES)
    contain physically impossible values (>1e300) from division-by-near-zero
    on instantaneous flows. These overflow float32 and corrupt any model.

    Caps are computed from TRAINING data only. Never fit caps on test data.

    Parameters
    ----------
    df : DataFrame
        Training data (typically UNSW).
    features : list of str
        Which columns to compute caps for.
    percentile : float
        The percentile of finite values to use as the cap.

    Returns
    -------
    dict mapping feature name -> cap value.
    """
    caps = {}
    for c in features:
        finite = df[c][np.isfinite(df[c])]
        caps[c] = float(np.percentile(finite, percentile))
    return caps


def apply_caps(df, caps, features=FEATURES):
    """Clip extreme and non-finite values using precomputed caps.

    Applies identically to both training and test sets. Cast to float32
    at the end so both datasets present with the same dtype to models.

    Parameters
    ----------
    df : DataFrame
    caps : dict
        Output of compute_caps().
    features : list of str

    Returns
    -------
    Cleaned copy of df.
    """
    df = df.copy()
    for c in features:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(caps[c])
        df[c] = df[c].clip(upper=caps[c])
    df[features] = df[features].astype("float32")
    return df


def load_and_clean(unsw_path, cic_path, cache_dir=None,
                   n_per_class=N_PER_CLASS, seed=DEFAULT_SEED):
    """Load both datasets, balance them, and clean the CIC overflow.

    Uses caches if present in cache_dir. Caps are always derived from UNSW
    (training) and applied identically to both datasets.

    Returns
    -------
    (df_unsw, df_cic) : tuple of cleaned DataFrames.
    """
    import os
    from .config import UNSW_BALANCED, CIC_BALANCED

    # Load or build UNSW.
    if cache_dir and os.path.exists(UNSW_BALANCED):
        df_unsw_raw = pd.read_parquet(UNSW_BALANCED)
    else:
        df_unsw_raw = load_balanced(unsw_path, n_per_class, seed)

    # Load or build CIC.
    if cache_dir and os.path.exists(CIC_BALANCED):
        df_cic_raw = pd.read_parquet(CIC_BALANCED)
    else:
        df_cic_raw = load_balanced(cic_path, n_per_class, seed)

    # Fit caps on UNSW only, apply to both.
    caps = compute_caps(df_unsw_raw)
    df_unsw = apply_caps(df_unsw_raw, caps)
    df_cic = apply_caps(df_cic_raw, caps)

    # Save cleaned versions.
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        df_unsw.to_parquet(UNSW_BALANCED, index=False)
        df_cic.to_parquet(CIC_BALANCED, index=False)

    return df_unsw, df_cic
