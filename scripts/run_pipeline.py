"""
End-to-end pipeline. Runs every experiment sequentially and saves all
outputs (models, results CSVs, figures) to OUTPUT_DIR.

Usage:
    python scripts/run_pipeline.py

Environment variables:
    IDS_BASE_DIR    - where raw datasets are (default ./data)
    IDS_OUTPUT_DIR  - where outputs go (default ./data/ids_output)

Expected wall-clock time on Colab T4/L4:
    ~5 min   data loading and cleaning (first run only, then cached)
    ~2 min   classical baselines
    ~10 min  feature ablation
    ~35 min  train RoBERTa on UNSW
    ~35 min  train RoBERTa on CIC
    ~5 min   adversarial evasion
    ~1 min   figure generation
    ---
    ~90 min  total end-to-end
"""

import os
import pandas as pd
import joblib

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    UNSW_PATH, CIC_PATH, OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    XGB_MODEL, LLM_ADAPTER_UNSW, LLM_ADAPTER_CIC,
    FEATURES, LABEL, DROPPED_ARTIFACTS, ADV_EPSILONS, DEFAULT_SEED,
    ensure_dirs,
)
from src.data_loading import load_and_clean
from src.evaluation import run_full_benchmark, evaluate
from src.models_classical import ablation_experiment
from src.models_llm import build_model_and_tokenizer, train_llm, timed_predict
from src.adversarial import build_evasion_suite, score_evasion_curve
from src.figures import (figure_1_three_axis, figure_2_transfer_matrix,
                         figure_3_evasion_curve, figure_4_ablation,
                         figure_5_speed_accuracy)

import xgboost as xgb


def main():
    ensure_dirs()

    # ------------------------------------------------------------------
    # Step 1: Load and clean data.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading and cleaning datasets")
    print("=" * 60)
    df_unsw, df_cic = load_and_clean(UNSW_PATH, CIC_PATH,
                                      cache_dir=OUTPUT_DIR)
    print(f"  UNSW: {len(df_unsw):,} rows")
    print(f"  CIC:  {len(df_cic):,} rows")

    # ------------------------------------------------------------------
    # Step 2: Feature ablation to justify feature set C.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Feature ablation")
    print("=" * 60)
    TTL = ["MIN_TTL", "MAX_TTL"]
    PKT = ["MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
           "SHORTEST_FLOW_PKT", "LONGEST_FLOW_PKT"]
    HIGH = ["SERVER_TCP_FLAGS", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]

    # Ablation needs the full 38-feature set to compare against.
    from src.config import FEATURES as F_LOCKED  # 29
    FULL_38 = F_LOCKED + DROPPED_ARTIFACTS  # 38, before locking
    feature_sets = {
        "A_drop_TTL_only":   [f for f in FULL_38 if f not in TTL],
        "B_drop_TTL_pktlen": [f for f in FULL_38 if f not in TTL + PKT],
        "C_drop_aggressive": [f for f in FULL_38
                              if f not in TTL + PKT + HIGH],
    }
    ablation = ablation_experiment(df_unsw, df_cic, feature_sets)
    ablation.to_csv(f"{RESULTS_DIR}/feature_ablation.csv", index=False)
    print(ablation)

    # ------------------------------------------------------------------
    # Step 3: Classical baseline through the harness.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Classical baseline (XGBoost, 4 seeds x 4 conditions)")
    print("=" * 60)
    xgb_results = run_full_benchmark(
        xgb.XGBClassifier,
        {"n_estimators": 100, "n_jobs": -1, "eval_metric": "logloss"},
        df_unsw, df_cic,
    )
    xgb_results.to_csv(f"{RESULTS_DIR}/xgb_results.csv", index=False)
    print(xgb_results.groupby("condition")["f1"].mean().round(4))

    # Save one XGBoost trained on the full UNSW for the evasion experiment.
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=DEFAULT_SEED,
                                   n_jobs=-1, eval_metric="logloss")
    from sklearn.model_selection import train_test_split
    tr_unsw, te_unsw = train_test_split(df_unsw, test_size=0.3,
                                          random_state=DEFAULT_SEED,
                                          stratify=df_unsw[LABEL])
    xgb_model.fit(tr_unsw[FEATURES], tr_unsw[LABEL])
    joblib.dump(xgb_model, XGB_MODEL)

    # ------------------------------------------------------------------
    # Step 4: LLM arm - train on UNSW.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4a: Fine-tune RoBERTa on UNSW")
    print("=" * 60)
    model_unsw, tokenizer = build_model_and_tokenizer()
    model_unsw, _ = train_llm(model_unsw, tokenizer, df_unsw,
                              output_dir=LLM_ADAPTER_UNSW)

    # Score UNSW-trained model on SD_unsw and CD_unsw2cic.
    val_unsw = te_unsw
    llm_results = []
    for df_test, cond in [(val_unsw, "SD_unsw"), (df_cic, "CD_unsw2cic")]:
        preds, elapsed, fps = timed_predict(model_unsw, tokenizer, df_test)
        r = evaluate(df_test[LABEL].values, preds,
                     model_name="RoBERTa-LoRA",
                     condition=cond, seed=DEFAULT_SEED)
        r["flows_sec"] = fps
        r["train_source"] = "UNSW"
        llm_results.append(r)

    # ------------------------------------------------------------------
    # Step 5: LLM arm - train on CIC (fresh model).
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4b: Fine-tune fresh RoBERTa on CIC")
    print("=" * 60)
    model_cic, tokenizer = build_model_and_tokenizer()
    model_cic, val_cic = train_llm(model_cic, tokenizer, df_cic,
                                    output_dir=LLM_ADAPTER_CIC)

    for df_test, cond in [(val_cic, "SD_cic"), (df_unsw, "CD_cic2unsw")]:
        preds, elapsed, fps = timed_predict(model_cic, tokenizer, df_test)
        r = evaluate(df_test[LABEL].values, preds,
                     model_name="RoBERTa-LoRA",
                     condition=cond, seed=DEFAULT_SEED)
        r["flows_sec"] = fps
        r["train_source"] = "CIC"
        llm_results.append(r)

    llm_df = pd.DataFrame(llm_results)
    llm_df.to_csv(f"{RESULTS_DIR}/llm_results.csv", index=False)

    # ------------------------------------------------------------------
    # Step 6: Adversarial evasion (same-dataset UNSW).
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Adversarial evasion")
    print("=" * 60)
    attacks_only = val_unsw[val_unsw[LABEL] == 1].copy()
    adv_datasets, perturbable = build_evasion_suite(
        tr_unsw, attacks_only, ADV_EPSILONS,
    )
    print(f"  Perturbable features: {len(perturbable)}")
    print(f"  Attack samples: {len(attacks_only):,}")

    xgb_evasion = score_evasion_curve(
        lambda df: xgb_model.predict(df[FEATURES]),
        adv_datasets, "XGBoost",
    )
    llm_evasion = score_evasion_curve(
        lambda df: timed_predict(model_unsw, tokenizer, df)[0],
        adv_datasets, "RoBERTa-LoRA",
    )
    evasion_df = pd.concat([xgb_evasion, llm_evasion], ignore_index=True)
    evasion_df.to_csv(f"{RESULTS_DIR}/evasion_curve.csv", index=False)

    # ------------------------------------------------------------------
    # Step 7: Figures.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Generating figures")
    print("=" * 60)
    xgb_mean = xgb_results.groupby("condition")["f1"].mean().round(4).to_dict()
    llm_mean = dict(zip(llm_df["condition"], llm_df["f1"]))

    figure_1_three_axis(xgb_mean, llm_mean, evasion_df,
                        save_path=f"{FIGURES_DIR}/fig1_three_axis_summary.png")
    figure_2_transfer_matrix(xgb_mean, llm_mean,
                             save_path=f"{FIGURES_DIR}/fig2_transfer_matrix.png")
    figure_3_evasion_curve(evasion_df,
                           save_path=f"{FIGURES_DIR}/fig3_evasion_curve.png")
    figure_4_ablation(ablation,
                      save_path=f"{FIGURES_DIR}/fig4_feature_ablation.png")

    # Speed comparison at SD_unsw.
    xgb_sp = int(xgb_results[xgb_results.condition == "SD_unsw"]
                 ["flows_sec"].mean())
    llm_sp = int(llm_df[llm_df.condition == "SD_unsw"]["flows_sec"].values[0])
    figure_5_speed_accuracy(xgb_sp, llm_sp,
                             xgb_mean["SD_unsw"], llm_mean["SD_unsw"],
                             save_path=f"{FIGURES_DIR}/fig5_speed_accuracy.png")

    print("\nDone. All outputs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
