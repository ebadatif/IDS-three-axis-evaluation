"""
Turn a numeric flow record into text an LLM can read.

Format:  FEATURE=value FEATURE=value ... FEATURE=value
    - Feature names retained so the LLM can leverage its pretrained
      understanding of tokens like "BYTES", "DURATION", "RETRANSMITTED".
    - Integer values stripped of trailing ".0" to save tokens.
    - Floats rounded to 2 decimal places (the 7th decimal is noise anyway).

At 29 features, each flow serializes to ~150-250 tokens with RoBERTa's BPE
tokenizer, comfortably below MAX_LEN=256.
"""

import torch
from torch.utils.data import Dataset as TorchDataset

from .config import FEATURES, LABEL, LLM_MAX_LEN


def flow_to_text(row, features=FEATURES):
    """Serialize one flow (a DataFrame row) into a compact text string."""
    parts = []
    for f in features:
        v = float(row[f])          # numpy floats need coercion for is_integer
        if v.is_integer():
            v = int(v)
        else:
            v = round(v, 2)
        parts.append(f"{f}={v}")
    return " ".join(parts)


class FlowDataset(TorchDataset):
    """PyTorch Dataset that tokenizes flows on demand.

    We use a plain torch Dataset rather than a HuggingFace `datasets.Dataset`
    with `.set_format("torch")` because the latter triggers a torchvision
    import (for VideoReader) that breaks on some Colab environments.

    Parameters
    ----------
    df : DataFrame with all columns in FEATURES plus LABEL.
    tokenizer : HF tokenizer.
    seed : int, for shuffling.
    n : int or None, subsample size.
    max_len : int, token truncation length.
    """

    def __init__(self, df, tokenizer, seed=42, n=None, max_len=LLM_MAX_LEN):
        d = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        if n:
            d = d.iloc[:n]
        self.texts = [flow_to_text(d.iloc[i]) for i in range(len(d))]
        self.labels = d[LABEL].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
