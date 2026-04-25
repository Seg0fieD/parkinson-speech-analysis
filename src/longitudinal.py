from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "longitudinal"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "longitudinal"

PRIMARY_MARKERS = {
    "jitter":   "jitterLocal_sma3nz_amean",
    "shimmer":  "shimmerLocaldB_sma3nz_amean",
    "hnr":      "HNRdBACF_sma3nz_amean",
    "f0_mean":  "F0semitoneFrom27.5Hz_sma3nz_amean",
    "f0_var":   "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness": "loudness_sma3_amean",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_longitudinal() -> pd.DataFrame:
    manifest = pd.read_csv(INTERIM / "manifest.csv")
    manifest = manifest[manifest["qc_pass"] & (manifest["group"] == "pd")].copy()

    n_sessions = manifest.groupby("subject_id")["session"].nunique()
    multi_ids = n_sessions[n_sessions > 1].index.tolist()
    df = manifest[manifest["subject_id"].isin(multi_ids)].copy()

    df["date"] = pd.to_datetime(df["date"])
    baseline = df.groupby("subject_id")["date"].transform("min")
    df["days_from_baseline"] = (df["date"] - baseline).dt.days
    df["months_from_baseline"] = df["days_from_baseline"] / 30.44

    egemaps = pd.read_parquet(PROCESSED / "features_egemaps.parquet")
    df = df.merge(egemaps, on="file_path", validate="1:1")

    return df


# ---------------------------------------------------------------------------
# Confound detection
# ---------------------------------------------------------------------------


def detect_session_confounds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in df.groupby("subject_id"):
        per_session = (
            sub.groupby("session")["sample_rate"]
            .agg(lambda s: sorted(set(s.tolist())))
        )
        sr_set = sorted({sr for srs in per_session for sr in srs})
        rows.append({
            "subject_id": sid,
            "n_sessions": int(sub["session"].nunique()),
            "sample_rates_per_session": per_session.to_dict(),
            "n_distinct_rates": len(sr_set),
            "consistent_equipment": len(sr_set) == 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MixedLM fitters
# ---------------------------------------------------------------------------


def _fit_mixedlm(df: pd.DataFrame, feature: str, marker_label: str,
                 task: str | None = None) -> dict:
    if task is not None:
        sub = df[df["task_code"] == task].copy()
        formula = f'Q("{feature}") ~ months_from_baseline'
    else:
        sub = df.copy()
        formula = f'Q("{feature}") ~ months_from_baseline + C(task_code)'

    sub = sub.dropna(subset=[feature, "months_from_baseline"])
    base = {
        "marker": marker_label, "task": task or "POOLED",
        "n_obs": len(sub),
        "n_subj": sub["subject_id"].nunique(),
    }
    if len(sub) < 4 or sub["subject_id"].nunique() < 2:
        return {**base, "beta_months": np.nan, "se": np.nan,
                "z": np.nan, "p": np.nan, "converged": False}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(formula, data=sub, groups=sub["subject_id"])
            res = md.fit(reml=True, method="lbfgs")
        return {
            **base,
            "beta_months": float(res.params.get("months_from_baseline", np.nan)),
            "se":          float(res.bse.get("months_from_baseline", np.nan)),
            "z":           float(res.tvalues.get("months_from_baseline", np.nan)),
            "p":           float(res.pvalues.get("months_from_baseline", np.nan)),
            "converged":   bool(res.converged),
        }
    except Exception as e:
        return {**base, "beta_months": np.nan, "se": np.nan,
                "z": np.nan, "p": np.nan, "converged": False, "error": str(e)[:80]}


def fit_pooled_models(df: pd.DataFrame, confounded: bool) -> pd.DataFrame:
    rows = [_fit_mixedlm(df, feat, lab) for lab, feat in PRIMARY_MARKERS.items()]
    out = pd.DataFrame(rows)
    out["equipment_confounded"] = confounded
    return out


def fit_per_task_models(df: pd.DataFrame, confounded: bool) -> pd.DataFrame:
    tasks = sorted(df["task_code"].unique())
    rows = []
    for task in tasks:
        for lab, feat in PRIMARY_MARKERS.items():
            rows.append(_fit_mixedlm(df, feat, lab, task=task))
    out = pd.DataFrame(rows)
    out["equipment_confounded"] = confounded
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_trajectories(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    sr_marker = {16000: "o", 44100: "s"}

    for ax, (label, col) in zip(axes.flat, PRIMARY_MARKERS.items()):
        agg = (
            df.groupby(["subject_id", "session", "months_from_baseline"])
            .agg(value=(col, "mean"), sample_rate=("sample_rate", "first"))
            .reset_index()
        )
        for sid, group in agg.groupby("subject_id"):
            group = group.sort_values("months_from_baseline")
            line, = ax.plot(group["months_from_baseline"], group["value"],
                            "-", label=sid.replace("pd_", ""))
            for _, row in group.iterrows():
                ax.plot(row["months_from_baseline"], row["value"],
                        marker=sr_marker.get(row["sample_rate"], "x"),
                        color=line.get_color(), markersize=8)
        ax.set_title(label)
        ax.set_xlabel("months from baseline")
        ax.set_ylabel("")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    sr_handles = [
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="", label="16 kHz"),
        plt.Line2D([0], [0], marker="s", color="gray", linestyle="", label="44.1 kHz"),
    ]
    fig.legend(
        handles + sr_handles,
        labels + ["16 kHz", "44.1 kHz"],
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.93),
        frameon=True,
    )
    fig.suptitle(
        "Longitudinal trajectories — primary markers (mean across tasks per session)\n"
        "Marker shape = original sample rate; shape changes ⇒ equipment switched",
        y=1.04, fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(FIG_DIR / "trajectories.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid")

    df = load_longitudinal()

    confound = detect_session_confounds(df)
    confound.to_csv(TABLES_DIR / "equipment_confound.csv", index=False)
    any_confounded = bool((~confound["consistent_equipment"]).any())

    pooled = fit_pooled_models(df, confounded=any_confounded)
    pooled.to_csv(TABLES_DIR / "pooled.csv", index=False)
    print("\nPooled MixedLM:")
    print(pooled.to_string(index=False))

    per_task = fit_per_task_models(df, confounded=any_confounded)
    per_task.to_csv(TABLES_DIR / "per_task.csv", index=False)

    plot_trajectories(df)

    print(f"\nTables: {TABLES_DIR.relative_to(PROJECT_ROOT)}/ | Figures: {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    if any_confounded:
        print("Note: equipment_confounded=True — see equipment_confound.csv")


if __name__ == "__main__":
    run()
