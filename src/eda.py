from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "eda"

GROUP_ORDER = ["young_hc", "elderly_hc", "pd"]
GROUP_PALETTE = {"young_hc": "#4C72B0", "elderly_hc": "#55A868", "pd": "#C44E52"}
COMPARABLE_TASKS = ["B1", "B2", "PR1"]

MARKER_COLS = {
    "Jitter (local)":      "jitterLocal_sma3nz_amean",
    "Shimmer (local dB)":  "shimmerLocaldB_sma3nz_amean",
    "HNR (dB)":            "HNRdBACF_sma3nz_amean",
    "F0 mean (semitones)": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0 var (stddevNorm)": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "Loudness mean":       "loudness_sma3_amean",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    manifest = pd.read_csv(INTERIM / "manifest.csv")
    manifest = manifest[manifest["qc_pass"]].reset_index(drop=True)

    egemaps = pd.read_parquet(PROCESSED / "features_egemaps.parquet")
    w2v2 = pd.read_parquet(PROCESSED / "features_w2v2.parquet")

    ege = manifest.merge(egemaps, on="file_path", validate="1:1")
    wv = manifest.merge(w2v2, on="file_path", validate="1:1")

    ege_cols = [c for c in egemaps.columns if c != "file_path"]
    wv_cols = [c for c in w2v2.columns if c != "file_path"]

    return ege, wv, ege_cols, wv_cols


# ---------------------------------------------------------------------------
# Cohort overview
# ---------------------------------------------------------------------------


def cohort_summary(manifest: pd.DataFrame) -> None:
    subj = manifest.drop_duplicates("subject_session_id")[
        ["subject_session_id", "group", "sex", "age"]
    ]
    print("\nSubjects per group:")
    print(subj["group"].value_counts().reindex(GROUP_ORDER))
    print("\nFiles per group x task:")
    print(
        manifest.groupby(["group", "task_code"]).size().unstack(fill_value=0)
        .reindex(GROUP_ORDER)
    )


# ---------------------------------------------------------------------------
# Age and sex confounds
# ---------------------------------------------------------------------------


def plot_age_sex(manifest: pd.DataFrame) -> None:
    subj = manifest.drop_duplicates("subject_session_id")[
        ["subject_session_id", "group", "sex", "age"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(
        data=subj, x="group", y="age", hue="group",
        order=GROUP_ORDER, palette=GROUP_PALETTE, legend=False, ax=axes[0],
    )
    sns.stripplot(
        data=subj, x="group", y="age",
        order=GROUP_ORDER, color="black", size=3, alpha=0.5, ax=axes[0],
    )
    axes[0].set_title("Age by group (per subject)")
    axes[0].set_xlabel("")

    sex_ct = (
        subj.groupby(["group", "sex"]).size().unstack(fill_value=0).reindex(GROUP_ORDER)
    )
    sex_ct.plot(kind="bar", stacked=True, ax=axes[1], color=["#DD8452", "#8172B3"])
    axes[1].set_title("Sex by group (per subject)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("count")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_age_sex_by_group.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dysphonia markers
# ---------------------------------------------------------------------------


def plot_markers(df: pd.DataFrame, title_suffix: str, fname: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (label, col) in zip(axes.flat, MARKER_COLS.items()):
        sns.boxplot(
            data=df, x="group", y=col, hue="group",
            order=GROUP_ORDER, palette=GROUP_PALETTE, legend=False,
            ax=ax, showfliers=False,
        )
        sns.stripplot(
            data=df, x="group", y=col,
            order=GROUP_ORDER, color="black", size=2, alpha=0.3, ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(f"Dysphonia markers — {title_suffix}", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_marker_by_task(ege: pd.DataFrame) -> None:
    task_order = sorted(ege["task_code"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for ax, (label, col) in zip(
        axes,
        [
            ("Jitter (local)", "jitterLocal_sma3nz_amean"),
            ("F0 var (stddevNorm)", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm"),
        ],
    ):
        sns.boxplot(
            data=ege, x="task_code", y=col, hue="group",
            order=task_order, hue_order=GROUP_ORDER,
            palette=GROUP_PALETTE, showfliers=False, ax=ax,
        )
        ax.set_title(f"{label} by task and group")
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_markers_by_task.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def fit_pca(df: pd.DataFrame, feat_cols: list[str], n: int = 2) -> tuple[np.ndarray, PCA]:
    X = df[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=np.nanmean(X))
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n, random_state=0)
    return pca.fit_transform(Xs), pca


def plot_2d(
    df: pd.DataFrame,
    Z: np.ndarray,
    title: str,
    fname: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for g in GROUP_ORDER:
        m = (df["group"] == g).to_numpy()
        axes[0].scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.6,
                        label=g, color=GROUP_PALETTE[g])
    axes[0].set_title(f"{title} — by group")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    axes[0].legend()

    tasks = sorted(df["task_code"].unique())
    cmap = plt.get_cmap("tab20", len(tasks))
    for i, t in enumerate(tasks):
        m = (df["task_code"] == t).to_numpy()
        axes[1].scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.6, label=t, color=cmap(i))
    axes[1].set_title(f"{title} — by task")
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel(ylabel)
    axes[1].legend(fontsize=7, ncol=2, loc="best")

    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=120)
    plt.close(fig)


def run_pca_panels(
    ege: pd.DataFrame, ege_cols: list[str],
    wv: pd.DataFrame, wv_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ege_cmp = ege[ege["task_code"].isin(COMPARABLE_TASKS)].reset_index(drop=True)
    wv_cmp = wv[wv["task_code"].isin(COMPARABLE_TASKS)].reset_index(drop=True)

    for df, cols, title, fname in [
        (ege_cmp, ege_cols, "eGeMAPS PCA (B1/B2/PR1)", "04a_pca_egemaps_comparable.png"),
        (ege,     ege_cols, "eGeMAPS PCA (all tasks)", "04b_pca_egemaps_all.png"),
        (wv_cmp,  wv_cols,  "wav2vec2 PCA (B1/B2/PR1)", "05a_pca_w2v2_comparable.png"),
        (wv,      wv_cols,  "wav2vec2 PCA (all tasks)", "05b_pca_w2v2_all.png"),
    ]:
        Z, pca = fit_pca(df, cols)
        xl = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        yl = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
        plot_2d(df, Z, title, fname, xl, yl)

    return ege_cmp, wv_cmp


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def run_umap_panels(
    ege: pd.DataFrame, ege_cols: list[str],
    wv: pd.DataFrame, wv_cols: list[str],
) -> None:
    try:
        import umap
    except ImportError:
        return

    def _fit(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        X = df[cols].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=np.nanmean(X))
        Xs = StandardScaler().fit_transform(X)
        return umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=0
        ).fit_transform(Xs)

    Z = _fit(ege, ege_cols)
    plot_2d(ege, Z, "eGeMAPS UMAP (all tasks)",
            "06_umap_egemaps_all.png", "UMAP-1", "UMAP-2")

    Z = _fit(wv, wv_cols)
    plot_2d(wv, Z, "wav2vec2 UMAP (all tasks)",
            "07_umap_w2v2_all.png", "UMAP-1", "UMAP-2")


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------


def silhouette_table(
    ege: pd.DataFrame, ege_cmp: pd.DataFrame, ege_cols: list[str],
    wv: pd.DataFrame, wv_cmp: pd.DataFrame, wv_cols: list[str],
) -> pd.DataFrame:
    def sil(df: pd.DataFrame, cols: list[str]) -> float:
        X = StandardScaler().fit_transform(
            np.nan_to_num(df[cols].to_numpy(dtype=np.float64))
        )
        return silhouette_score(
            X, df["group"].to_numpy(),
            sample_size=min(2000, len(df)), random_state=0,
        )

    rows = []
    for label, df in [("comparable (B1/B2/PR1)", ege_cmp), ("all tasks", ege)]:
        rows.append({"feature_set": "eGeMAPS", "view": label,
                     "silhouette(group)": round(sil(df, ege_cols), 4)})
    for label, df in [("comparable (B1/B2/PR1)", wv_cmp), ("all tasks", wv)]:
        rows.append({"feature_set": "wav2vec2", "view": label,
                     "silhouette(group)": round(sil(df, wv_cols), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid")

    ege, wv, ege_cols, wv_cols = load_data()

    cohort_summary(ege)
    plot_age_sex(ege)

    plot_markers(
        ege[ege["task_code"].isin(COMPARABLE_TASKS)],
        "B1, B2, PR1 only (cross-group comparable)",
        "02a_markers_comparable_tasks.png",
    )
    plot_markers(ege, "all tasks (young_hc has only B1/B2/PR1)",
                 "02b_markers_all_tasks.png")
    plot_marker_by_task(ege)

    ege_cmp, wv_cmp = run_pca_panels(ege, ege_cols, wv, wv_cols)
    run_umap_panels(ege, ege_cols, wv, wv_cols)

    sil_df = silhouette_table(ege, ege_cmp, ege_cols, wv, wv_cmp, wv_cols)
    print("\nSilhouette of group labels:")
    print(sil_df.to_string(index=False))

    print(f"\nFigures: {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    run()
