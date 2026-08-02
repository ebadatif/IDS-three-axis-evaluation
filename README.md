# IDS Three-Axis Evaluation

Cross-dataset and adversarial evaluation of RoBERTa-LoRA vs XGBoost for network intrusion detection.

## Headline finding

Same-dataset evaluation, cross-dataset transfer, and adversarial evasion give **three different verdicts** on the same models. No universal winner.

| Axis | XGBoost | RoBERTa-LoRA | Winner |
|---|---|---|---|
| Same-dataset (UNSW-NB15) | 0.997 | 0.995 | Tie |
| Same-dataset (CIC-IDS2018) | 0.978 | 0.975 | Tie |
| Cross-dataset (UNSW → CIC) | **0.812** | 0.621 | XGBoost (+19 pts) |
| Cross-dataset (CIC → UNSW) | 0.073 | 0.038 | Both collapse |
| Adversarial evasion (ε=0.5) | 0.632 | 0.898 | RoBERTa (+27 pts) |
| Adversarial evasion (ε=1.0) | 0.000 | **0.842** | RoBERTa (+84 pts) |

Which evaluation you run determines your conclusion about LLM competitiveness. The appropriate model depends on the deployment threat model: distribution shift → XGBoost, adaptive adversary → RoBERTa.

## Repository structure

```
ids-three-axis-eval/
├── src/                        # Reusable Python modules
│   ├── config.py               # Paths, feature list, hyperparameters
│   ├── data_loading.py         # Streaming loader, overflow fix
│   ├── flow_serialization.py   # Flow → text for the LLM
│   ├── evaluation.py           # Fair evaluation harness
│   ├── models_classical.py     # RF, XGBoost, feature ablation
│   ├── models_llm.py           # RoBERTa + LoRA training and inference
│   ├── adversarial.py          # Feature-space evasion attack
│   └── figures.py              # All five paper figures
├── scripts/
│   └── run_pipeline.py         # End-to-end runner
├── notebooks/
│   └── main_pipeline.ipynb     # Original exploratory notebook
├── data/                       # (Download datasets here)
├── models/                     # Trained models saved here
├── results/                    # CSV outputs saved here
├── figures/                    # Paper figures saved here
├── docs/
│   └── RESEARCH_LOG.md         # Decisions, findings, reasoning
├── requirements.txt
└── LICENSE
```

## Setup

```bash
git clone https://github.com/<your-user>/ids-three-axis-eval.git
cd ids-three-axis-eval
pip install -r requirements.txt
```

**Do not install `bitsandbytes`.** It's not needed (we use fp16 + LoRA, not 4-bit quantization) and its presence causes GPU dependency conflicts on Colab.

## Data

Download the NetFlow v2 collection by Sarhan, Layeghy, Portmann:
- `NF-UNSW-NB15-v2.parquet` (~50 MB)
- `NF-CSE-CIC-IDS2018-v2.parquet` (~600 MB)

Available at: <https://staff.itee.uq.edu.au/marius/NIDS_datasets>

Place both files in `./data/` (or set the `IDS_BASE_DIR` environment variable).

## Reproduce

**End-to-end (about 90 min on a Colab T4/L4 GPU):**

```bash
python scripts/run_pipeline.py
```

This runs data loading → feature ablation → classical baseline (4 seeds × 4 conditions) → LLM training in both directions → adversarial evasion → figure generation. All outputs land in `./data/ids_output/`.

**Component by component (recommended for exploration):**

```python
from src.data_loading import load_and_clean
from src.evaluation import run_full_benchmark
from src.models_classical import ablation_experiment
import xgboost as xgb

# 1. Load and clean.
df_unsw, df_cic = load_and_clean(unsw_path, cic_path, cache_dir="./out")

# 2. Fair evaluation of any model across all four conditions.
results = run_full_benchmark(
    xgb.XGBClassifier,
    {"n_estimators": 100, "n_jobs": -1, "eval_metric": "logloss"},
    df_unsw, df_cic,
)
```

## Method summary

**Datasets.** NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 - both re-extracted from raw PCAPs into an identical 43-feature NetFlow schema, which is what makes fair cross-dataset comparison possible.

**Feature set.** Locked to 29 features via cross-dataset ablation (see `figures/fig4_feature_ablation.png`). Dropped: bias-prone identifiers (ports, DNS query ID) and cross-dataset-verified artifacts (TTL, packet length, TCP window/flags).

**Classical arm.** XGBoost, 100 trees, 4 seeds. Random Forest as secondary baseline.

**LLM arm.** RoBERTa-base, LoRA adapters (r=8, α=16, target = query/key/value projections), ~0.82% of parameters trainable. Flow records serialized as `FEATURE=value ...` text.

**Adversarial attack.** Feature-space "drift toward benign": linear interpolation between attack flows and the benign centroid at 10 perturbation strengths. Domain constraints enforced (non-negativity, integer counts, retransmitted ≤ total). Fixed categorical features (protocol, ICMP type) not perturbed. Same-dataset only.

**Evaluation.** Every model scored by the same `evaluate()` function - F1, precision, recall, balanced accuracy, FPR, FNR, flows/sec.

## Limitations

- Single seed for LLM experiments (significance testing incomplete).
- Feature-space adversarial only - problem-space realizability not guaranteed.
- Two datasets only; generalization to other networks not tested.
- One LLM architecture (RoBERTa-base); other encoders may behave differently.
- Text serialization gives the LLM structural anchors (feature names) that numeric-only classical models lack.

## Citation

If you use this code or the findings, please cite:

```bibtex
@misc{atif2026ids,
  title = {Same Benchmark, Different Verdict: A Three-Axis Stress Test of LLM
           vs Classical ML for Network Intrusion Detection under Distribution
           Shift and Adversarial Evasion},
  author = {Atif, Muhammad Ebad and Ali, Muhammad Haider},
  year = {2026},
  note = {arXiv preprint (forthcoming)},
  url = {https://github.com/Nukka2005/IDS-three-axis-evaluation}
}
```

## License

MIT - see `LICENSE`.
