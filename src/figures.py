"""
Paper figures. Each function is standalone and takes results DataFrames
as input, so figures can be regenerated without re-running experiments.

Five figures:
  1. Three-axis summary (visual abstract)
  2. Cross-dataset transfer matrix heatmap
  3. Adversarial evasion curves
  4. Feature ablation
  5. Speed vs accuracy tradeoff
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Publication-quality defaults.
_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
XGB_COLOR = "#1976D2"      # blue
LLM_COLOR = "#E64A19"      # orange-red
GREY = "#666666"


def _apply_style():
    mpl.rcParams.update(_STYLE)


def figure_1_three_axis(xgb_mean, llm_mean, evasion_df, save_path=None):
    """The visual abstract - one image capturing the whole finding.

    Bars for same-dataset UNSW, same-dataset CIC, cross-dataset UNSW->CIC,
    and adversarial evasion (at epsilon=0.5). Winner labeled per axis.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    axes_labels = [
        "Same-dataset\n(UNSW)\n\nTie",
        "Same-dataset\n(CIC)\n\nTie",
        "Cross-dataset\n(UNSW→CIC)\n\nWinner: XGBoost",
        "Adversarial\n(ε=0.5)\n\nWinner: RoBERTa",
    ]

    adv_xgb = evasion_df[(evasion_df.model == "XGBoost") &
                         (evasion_df.epsilon == 0.5)]["detection_rate"].values[0]
    adv_llm = evasion_df[(evasion_df.model == "RoBERTa-LoRA") &
                         (evasion_df.epsilon == 0.5)]["detection_rate"].values[0]

    xgb_bars = [xgb_mean["SD_unsw"], xgb_mean["SD_cic"],
                xgb_mean["CD_unsw2cic"], adv_xgb]
    llm_bars = [llm_mean["SD_unsw"], llm_mean["SD_cic"],
                llm_mean["CD_unsw2cic"], adv_llm]

    x = np.arange(len(axes_labels))
    w = 0.36
    b1 = ax.bar(x - w/2, xgb_bars, w, color=XGB_COLOR, label="XGBoost",
                edgecolor="white")
    b2 = ax.bar(x + w/2, llm_bars, w, color=LLM_COLOR, label="RoBERTa-LoRA",
                edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_ylabel("F1 / Detection rate")
    ax.set_title("Three axes, three verdicts - no universal winner",
                 fontsize=14, pad=12)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def figure_2_transfer_matrix(xgb_mean, llm_mean, save_path=None):
    """2x2 heatmap per model showing train-source x test-source F1.

    The off-diagonal cells are cross-dataset; the diagonal is same-dataset.
    Reveals the violent CIC->UNSW asymmetry across BOTH model families.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    def matrix(m):
        return np.array([
            [m["SD_unsw"],     m["CD_unsw2cic"]],
            [m["CD_cic2unsw"], m["SD_cic"]],
        ])

    for ax, mat, title in [(axes[0], matrix(xgb_mean), "XGBoost"),
                           (axes[1], matrix(llm_mean), "RoBERTa-LoRA")]:
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["UNSW", "CIC"])
        ax.set_yticklabels(["UNSW", "CIC"])
        ax.set_xlabel("Test on →")
        ax.set_ylabel("← Trained on")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] < 0.4 or mat[i, j] > 0.85 else "black"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color=color, fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=axes, shrink=0.7, label="F1")
    fig.suptitle("Cross-dataset transfer is violently asymmetric - "
                 "for both models", y=1.02, fontsize=13)

    if save_path:
        fig.savefig(save_path)
    return fig


def figure_3_evasion_curve(evasion_df, save_path=None):
    """Detection rate vs perturbation strength for both models."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for mname, color, marker in [("XGBoost", XGB_COLOR, "o"),
                                  ("RoBERTa-LoRA", LLM_COLOR, "s")]:
        sub = evasion_df[evasion_df.model == mname].sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["detection_rate"],
                color=color, marker=marker, linewidth=2.5, markersize=9,
                label=mname, zorder=3)

    ax.axhline(y=0.5, color=GREY, linestyle="--", alpha=0.6,
               label="50% threshold", zorder=1)
    ax.fill_between([-0.02, 1.02], 0, 0.5, color="red", alpha=0.06, zorder=0)

    ax.set_xlabel("Perturbation strength (ε)")
    ax.set_ylabel("Detection rate")
    ax.set_title("Adversarial evasion - the ranking reverses")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)

    if save_path:
        fig.savefig(save_path)
    return fig


def figure_4_ablation(ablation_df, save_path=None):
    """SD vs CD F1 across three feature-set hypotheses (A, B, C)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = [
        "A: drop TTL only\n(36 features)",
        "B: + drop pkt-length\n(32 features)",
        "C: + drop window/flags\n(29 features)",
    ]
    sd_vals = ablation_df["SD_f1"].values
    cd_vals = ablation_df["CD_f1"].values

    x = np.arange(len(labels)); w = 0.36
    b1 = ax.bar(x - w/2, sd_vals, w, color=GREY,
                label="Same-dataset F1", edgecolor="white")
    b2 = ax.bar(x + w/2, cd_vals, w, color=XGB_COLOR,
                label="Cross-dataset F1", edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_title("Same-dataset F1 is uninformative - "
                 "cross-dataset reveals which features are artifacts")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="gold", alpha=0.12, zorder=0)
    ax.annotate("LOCKED\nfeature set", xy=(x[-1] + 0.2, 0.822),
                xytext=(x[-1] + 0.55, 0.55),
                fontsize=11, fontweight="bold", color="darkgoldenrod",
                arrowprops=dict(arrowstyle="->", color="darkgoldenrod", lw=2),
                annotation_clip=False)

    if save_path:
        fig.savefig(save_path)
    return fig


def figure_5_speed_accuracy(xgb_speed, llm_speed, xgb_acc, llm_acc,
                             save_path=None):
    """The Mehavilla-style efficiency dimension."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter([xgb_speed], [xgb_acc], s=350, color=XGB_COLOR,
               label="XGBoost", edgecolor="white", linewidth=2, zorder=3)
    ax.scatter([llm_speed], [llm_acc], s=350, color=LLM_COLOR,
               label="RoBERTa-LoRA", edgecolor="white", linewidth=2, zorder=3)

    ax.annotate(f"XGBoost\n{xgb_speed:,} flows/sec\nF1 = {xgb_acc:.3f}",
                xy=(xgb_speed, xgb_acc), xytext=(20, -30),
                textcoords="offset points", fontsize=10,
                color=XGB_COLOR, fontweight="bold")
    ax.annotate(f"RoBERTa\n{llm_speed:,} flows/sec\nF1 = {llm_acc:.3f}",
                xy=(llm_speed, llm_acc), xytext=(20, 15),
                textcoords="offset points", fontsize=10,
                color=LLM_COLOR, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Inference speed (flows/sec, log scale)")
    ax.set_ylabel("Same-dataset F1")
    ax.set_title(f"Similar accuracy, ~{xgb_speed // llm_speed}× speed gap")
    ax.set_ylim(0.95, 1.005)
    ax.set_xlim(50, xgb_speed * 3)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both"); ax.set_axisbelow(True)

    if save_path:
        fig.savefig(save_path)
    return fig
