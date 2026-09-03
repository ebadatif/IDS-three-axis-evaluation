# Intrusion Detection System (IDS) Three-Axis Evaluation

Cross-dataset and adversarial evaluation of RoBERTa-LoRA vs XGBoost for
network intrusion detection.

Companion code for: *"Same Benchmark, Different Verdict: A Three-Axis
Stress Test of LLM vs Classical ML for Network Intrusion Detection under
Distribution Shift and Adversarial Evasion"* (Atif & Ali, 2026).

## Headline finding

Same-dataset evaluation, cross-dataset transfer, and adversarial evasion
give **three different verdicts** on the same two models. No universal
winner.

| Axis | XGBoost | RoBERTa-LoRA | Winner |
|---|---|---|---|
| Same-dataset (UNSW-NB15) | 0.9966 | 0.9938 | Tie |
| Same-dataset (CIC-IDS2018) | 0.9776 | 0.9745 | Tie |
| Cross-dataset (UNSW → CIC) | **0.8116** | 0.6582 | XGBoost (+15.3 pts) |
| Cross-dataset (CIC → UNSW) | 0.0730 | 0.0599 | Both collapse to chance |
| Adversarial evasion (ε=0.5) | 0.7720 | **0.9448** | RoBERTa (+17.3 pts) |

A second finding cuts across two of the three axes: **F1 alone hides
what's actually happening.** On the cross-dataset axis, RoBERTa's F1 of
0.6582 looks like moderate transfer, but its false positive rate is
0.7756 and its balanced accuracy is 0.5477 — barely above chance. It
sustains recall mainly by flagging most benign CIC traffic as an attack.
On the adversarial axis, the same FPR check confirms RoBERTa's
robustness is genuine (FPR stays at 0.0081 throughout the sweep, never
trading false alarms for detection). The identical diagnostic returns
opposite verdicts for the same model on two different axes — which is
why we report FPR alongside F1 throughout.

Note: the adversarial ε=1.0 endpoint is **excluded** from the headline
table above. At ε=1.0 the interpolation attack collapses every attack
flow to the same point in feature space (1 distinct row out of 15,000),
so it measures one model decision, not 15,000 independent trials. See
the paper's Methodology section and `docs/RESEARCH_LOG.md` for the full
degeneracy analysis; the full ten-point sweep including this endpoint is
in `results/block7_evasion_curve.csv` and Table VII of the paper's
appendix.

Which evaluation you run determines your conclusion about LLM
competitiveness. The right model choice depends on the deployment
threat model: distribution shift across networks → XGBoost; an adaptive
adversary who can manipulate flow features → RoBERTa, inference-budget
permitting (RoBERTa is roughly 5,800× slower at inference).

## Repository structure
ids-three-axis-eval/
├── src/ # Reusable Python modules
│ ├── config.py # Paths, feature list, hyperparameters
│ ├── data_loading.py # Streaming loader, overflow fix
│ ├── flow_serialization.py # Flow → text for the LLM
│ ├── evaluation.py # Fair evaluation harness
│ ├── models_classical.py # RF, XGBoost, feature ablation
│ ├── models_llm.py # RoBERTa + LoRA training and inference
│ ├── adversarial.py # Feature-space evasion attack
│ └── figures.py # All five paper figures
├── scripts/
│ └── run_pipeline.py # End-to-end runner
├── notebooks/
│ └── main_pipeline.ipynb # Original exploratory notebook
├── data/ # (Download datasets here)
├── models/ # Trained models saved here
├── results/ # CSV outputs saved here
├── figures/ # Paper figures saved here
├── docs/
│ └── RESEARCH_LOG.md # Decisions, findings, reasoning
├── requirements.txt
└── LICENSE


## Setup

```bash
git clone https://github.com/ebadatif/IDS-three-axis-evaluation.git
cd IDS-three-axis-evaluation
pip install -r requirements.txt
```

**Do not install `bitsandbytes`.** It's not needed (we use fp16 + LoRA,
not 4-bit quantization) and its presence causes GPU dependency conflicts
on Colab.

## Data

Download the NetFlow v2 collection by Sarhan, Layeghy, and Portmann:
- `NF-UNSW-NB15-v2.parquet` (~50 MB)
- `NF-CSE-CIC-IDS2018-v2.parquet` (~600 MB)

Available at: <https://staff.itee.uq.edu.au/marius/NIDS_datasets>

Place both files in `./data/` (or set the `IDS_BASE_DIR` environment
variable).

## Reproduce

**End-to-end (about 90 minutes on a Colab T4/L4 GPU):**

```bash
python scripts/run_pipeline.py
```

This runs data loading → feature ablation → classical baseline (4 seeds
× 4 conditions) → LLM training in both directions, each from a fresh
pretrained checkpoint → adversarial evasion (with benign flows included
in the eval set) → figure generation. All outputs land in
`./data/ids_output/`.

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

If you re-run the harness, **clear any cached result CSVs first**
(`results/block4_harness_*.csv`). The pipeline loads cached results when
present rather than retraining, which is fast for iteration but means a
stale cache from an earlier pipeline state will silently override a
fresh run.

## Method summary

**Datasets.** NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2, both
re-extracted from raw PCAPs into an identical 43-feature NetFlow schema
— the shared schema is what makes fair cross-dataset comparison
possible in the first place. Balanced to 50,000 flows per class in
every split to isolate distribution shift from class-prevalence shift
(raw prevalence differs threefold between the two networks).

**Feature set.** Locked to 29 features via a staged, three-set
cross-dataset ablation (`figures/fig4_feature_ablation.png`,
`results/block3_8_clean_ablation.csv`). The result is non-monotonic:
dropping TTL alone (Set A, 36 features) gives cross-dataset F1 = 0.0488;
also dropping packet-length features (Set B, 32 features) makes it
*worse* (F1 = 0.0282), because packet length carries both leakage and a
genuine transferable signal that Set B discards along with the leakage.
Only the aggressive Set C (29 features, additionally dropping
`SERVER_TCP_FLAGS` and TCP window fields) recovers transfer performance
(F1 = 0.8116). Same-dataset F1 barely moves across all three sets
(0.9966–0.9972), which is itself the point: same-dataset accuracy cannot
tell you which features are environmental artifacts.

**Classical arm.** XGBoost, 100 trees, 4 seeds (42–45). Random Forest as
a secondary baseline to check findings aren't gradient-boosting-specific.

**LLM arm.** RoBERTa-base, LoRA adapters (r=8, α=16, target =
query/key/value projections), ~0.82% of parameters trainable. Each
direction (UNSW-trained, CIC-trained) is fine-tuned in a single run from
a freshly loaded pretrained checkpoint with a freshly initialized
adapter — neither run carries optimization history from the other. Flow
records are serialized as `FEATURE=value ...` text before tokenization.

**Adversarial attack.** Feature-space "drift toward benign": linear
interpolation between each attack flow and the benign centroid
(computed from the training split only) at 10 perturbation strengths.
Domain constraints enforced: non-negativity, integer packet/byte counts,
retransmitted bytes bounded above by total bytes, fixed categorical
features (protocol, L7 protocol, ICMP type, DNS query type, FTP return
code) left unperturbed. The evaluation set at every ε includes both the
15,000 perturbed attack flows *and* 15,000 untouched benign flows —
benign traffic is never perturbed, which is what makes FPR measurable
and lets us distinguish genuine robustness from a model that simply
flags everything. Same-dataset (UNSW) only.

**Evaluation.** Every model, every condition, scored by the same
`evaluate()` function: F1, precision, recall, balanced accuracy, FPR,
FNR, throughput (flows/sec).

## Limitations

- Single seed for LLM experiments; classical models use 4 seeds with
  meaningfully different cross-seed variance (zero for XGBoost, 0.083
  std for Random Forest), so this asymmetry matters for how confidently
  the LLM numbers can be compared.
- RoBERTa checkpoints are selected via F1 on a held-out subset that
  overlaps with the same-dataset reporting split; classical models use
  no validation-based selection, so LLM same-dataset/adversarial figures
  may be marginally optimistic relative to the classical arm.
- Feature-space adversarial attack only — perturbed flows have plausible
  *features*, but problem-space realizability (an actual attacker
  producing matching raw traffic) is not verified.
- The interpolation target is a single global benign centroid, which
  degenerates entirely at ε=1.0 (see Headline finding above). A
  per-flow or cluster-targeted attack variant was not tested.
- Two datasets only; the pipeline records only the binary label, so the
  volumetric-vs-behavioral hypothesis explaining the cross-dataset
  transfer asymmetry is plausible but not directly verified against
  per-attack-subtype recall.
- One LLM architecture (RoBERTa-base); other encoders may behave
  differently, particularly on the adversarial axis, where our leading
  explanation for RoBERTa's stability is architecture- and
  serialization-dependent by hypothesis.
- Text serialization gives the LLM explicit feature-name anchors that
  the numeric-only classical models never receive at inference time.

## Citation

If you use this code or the findings, please cite:

```bibtex
@misc{atif2026ids,
  title         = {Same Benchmark, Different Verdict: A Three-Axis Stress
                   Test of LLM vs Classical ML for Network Intrusion
                   Detection under Distribution Shift and Adversarial
                   Evasion},
  author        = {Atif, Muhammad Ebad and Ali, Muhammad Haider},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://github.com/ebadatif/IDS-three-axis-evaluation}
}
```

*(arXiv ID above is a placeholder —  will be updating it once the preprint is posted.)*

## License

MIT — see `LICENSE`.
