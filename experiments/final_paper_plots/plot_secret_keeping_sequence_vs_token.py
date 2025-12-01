import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_secret_keeping_results import (
    CHOSEN_GENDER_PROMPT,
    CHOSEN_TABOO_PROMPT,
    FONT_SIZE_BAR_VALUE,
    FONT_SIZE_LEGEND,
    FONT_SIZE_SUBPLOT_TITLE,
    FONT_SIZE_Y_AXIS_LABEL,
    FONT_SIZE_Y_AXIS_TICK,
    GENDER_HIGHLIGHT,
    SSC_HIGHLIGHT,
    TABOO_HIGHLIGHT,
    GENDER_JSON_DIR,
    SSC_JSON_DIR,
    TABOO_JSON_DIR,
    ci95,
    load_gender_results,
    load_ssc_results,
    load_taboo_results,
)
from shared_color_mapping import get_shared_palette

HATCH = "////"

IMAGE_FOLDER = "images"
SEQUENCE_VS_TOKEN_FOLDER = f"{IMAGE_FOLDER}/sequence_vs_token"
os.makedirs(SEQUENCE_VS_TOKEN_FOLDER, exist_ok=True)
OUTPUT_PATH = f"{SEQUENCE_VS_TOKEN_FOLDER}/secret_keeping_sequence_vs_token.png"


def _highlight_mean(results_by_lora: dict, keyword: str) -> tuple[float, float]:
    matches = [name for name in results_by_lora if name and keyword in name]
    assert len(matches) == 1, f"Expected one match for {keyword}, found {matches}"
    values = results_by_lora[matches[0]]
    mean = float(np.mean(values))
    return mean, ci95(values)


def _taboo_stats() -> tuple[str, float, float, float, float]:
    token_results, _ = load_taboo_results(
        TABOO_JSON_DIR, required_verbalizer_prompt=CHOSEN_TABOO_PROMPT, sequence=False
    )
    seq_results, _ = load_taboo_results(TABOO_JSON_DIR, required_verbalizer_prompt=CHOSEN_TABOO_PROMPT, sequence=True)
    token_mean, token_ci = _highlight_mean(token_results, TABOO_HIGHLIGHT)
    seq_mean, seq_ci = _highlight_mean(seq_results, TABOO_HIGHLIGHT)
    return ("Taboo (Gemma-2-9B-IT)", token_mean, token_ci, seq_mean, seq_ci)


def _gender_stats() -> tuple[str, float, float, float, float]:
    token_results, _ = load_gender_results(
        GENDER_JSON_DIR, required_verbalizer_prompt=CHOSEN_GENDER_PROMPT, sequence=False
    )
    seq_results, _ = load_gender_results(
        GENDER_JSON_DIR, required_verbalizer_prompt=CHOSEN_GENDER_PROMPT, sequence=True
    )
    token_mean, token_ci = _highlight_mean(token_results, GENDER_HIGHLIGHT)
    seq_mean, seq_ci = _highlight_mean(seq_results, GENDER_HIGHLIGHT)
    return ("Gender (Gemma-2-9B-IT)", token_mean, token_ci, seq_mean, seq_ci)


async def _ssc_stats() -> tuple[str, float, float, float, float]:
    token_results, _ = await load_ssc_results(SSC_JSON_DIR, sequence=False)
    seq_results, _ = await load_ssc_results(SSC_JSON_DIR, sequence=True)
    token_mean, token_ci = _highlight_mean(token_results, SSC_HIGHLIGHT)
    seq_mean, seq_ci = _highlight_mean(seq_results, SSC_HIGHLIGHT)
    return ("SSC (Llama-3.3-70B)", token_mean, token_ci, seq_mean, seq_ci)


def _annotate_bars(ax, bars, means, cis):
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ci + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BAR_VALUE,
        )


def plot_sequence_vs_token(stats: list[tuple[str, float, float, float, float]]):
    labels = [s[0] for s in stats]
    palette = get_shared_palette(labels)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(stats))
    width = 0.35

    token_means = [s[1] for s in stats]
    token_cis = [s[2] for s in stats]
    seq_means = [s[3] for s in stats]
    seq_cis = [s[4] for s in stats]
    colors = [palette[label] for label in labels]

    token_bars = ax.bar(
        x - width / 2.0, token_means, width, color=colors, yerr=token_cis, capsize=5, error_kw={"linewidth": 2}
    )
    seq_bars = ax.bar(
        x + width / 2.0,
        seq_means,
        width,
        color=colors,
        yerr=seq_cis,
        capsize=5,
        error_kw={"linewidth": 2},
        hatch=HATCH,
        edgecolor="black",
        linewidth=1.5,
    )

    _annotate_bars(ax, token_bars, token_means, token_cis)
    _annotate_bars(ax, seq_bars, seq_means, seq_cis)

    ax.set_title("Secret Keeping: Token vs. Sequence", fontsize=FONT_SIZE_SUBPLOT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_Y_AXIS_TICK)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_Y_AXIS_TICK)
    ax.set_ylabel("Average Accuracy", fontsize=FONT_SIZE_Y_AXIS_LABEL)

    dataset_handles = [Patch(facecolor=palette[label], edgecolor="black", label=label) for label in labels]
    token_handle = Patch(facecolor="white", edgecolor="black", label="Single Token")
    sequence_handle = Patch(facecolor="white", edgecolor="black", hatch=HATCH, label="Full Sequence")
    fig.legend(
        handles=dataset_handles + [token_handle, sequence_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE_LEGEND,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


async def main():
    stats = [
        _taboo_stats(),
        _gender_stats(),
        await _ssc_stats(),
    ]
    plot_sequence_vs_token(stats)


if __name__ == "__main__":
    asyncio.run(main())
