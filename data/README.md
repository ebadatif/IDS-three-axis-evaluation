# Data directory

This directory should contain the NetFlow v2 dataset files. **They are not committed to the repo** (too large).

## Required files

- `NF-UNSW-NB15-V2.parquet` (~50 MB, 1.99M rows)
- `NF-CSE-CIC-IDS2018-V2.parquet` (~600 MB, 17.1M rows)

## Download source

Both datasets are from the University of Queensland NetFlow v2 collection:

**<https://staff.itee.uq.edu.au/marius/NIDS_datasets>**

Alternative mirror (cleaned Parquet): search Kaggle for "NetFlow v2 NIDS datasets".

## Citation for the datasets

If you use these datasets, cite the original NetFlow v2 paper:

Sarhan, M., Layeghy, S., & Portmann, M. (2021). *Towards a standard feature set for network intrusion detection system datasets.* Mobile Networks and Applications.

## Verification

After downloading, run the schema check:

```python
from src.data_loading import load_balanced
df = load_balanced("./data/NF-UNSW-NB15-V2.parquet", n_per_class=100)
print(df.columns.tolist())  # Should show 43 columns
```

Both parquet files must have the same 43-column schema (they do, by design of the NetFlow v2 collection). If a column mismatch is reported, the download is corrupted.
