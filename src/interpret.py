from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "interpret"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "interpret"
STATS_DIR = PROJECT_ROOT / "reports" / "tables" / "stats"

N_FOLDS = 5
N_BOOT = 1000
RANDOM_STATE = 0
TOP_K = 15
PER_TASK_TASK = "VA1"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(INTERIM / "manifest.csv")
    manifest = manifest[
        manifest["qc_pass"]
        & (manifest["session"] == 1)
        & (manifest["group"].isin(["pd", "elderly_hc"]))
    ].copy()
    manifest["y"] = (manifest["group"] == "pd").astype(int)

    egemaps = pd.read_parquet(PROCESSED / "features_egemaps.parquet")
    df = manifest.merge(egemaps, on="file_path", validate="1:1")

    feat_cols = [c for c in egemaps.columns if c != "file_path"]
    return df, feat_cols


# ---------------------------------------------------------------------------
# Out-of-fold SHAP
# ---------------------------------------------------------------------------


def _shap_values_class1(explainer, X_scaled: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(X_scaled)
    if isinstance(sv, list):
        return sv[1]
    if hasattr(sv, "values"):
        sv = sv.values
    if sv.ndim == 3:
        return sv[..., 1]
    return sv


def oof_shap(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    feat_names: list[str], clf_factory, label: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sv_list: list[np.ndarray] = []
    xs_list: list[np.ndarray] = []

    splits = list(cv.split(X, y, groups))
    for fold, (tr, te) in enumerate(tqdm(splits, desc=f"SHAP {label}")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = StandardScaler().fit(X[tr])
            xtr = scaler.transform(X[tr])
            xte = scaler.transform(X[te])
            clf = clf_factory()
            clf.fit(xtr, y[tr])
            explainer = shap.TreeExplainer(clf)
            sv = _shap_values_class1(explainer, xte)
        sv_list.append(sv)
        xs_list.append(xte)

    sv_all = np.vstack(sv_list)
    xs_all = np.vstack(xs_list)

    summary = pd.DataFrame({
        "feature":          feat_names,
        "mean_abs_shap":    np.abs(sv_all).mean(axis=0),
        "mean_signed_shap": sv_all.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary, sv_all, xs_all


def plot_shap_summary(
    sv: np.ndarray, X_scaled: np.ndarray, feat_names: list[str],
    title: str, fpath: Path, top_k: int = TOP_K,
) -> None:
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        sv, X_scaled, feature_names=feat_names, max_display=top_k, show=False,
    )
    plt.title(title, fontsize=11, pad=15)
    plt.tight_layout()
    plt.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()


def plot_shap_bar(
    summary: pd.DataFrame, title: str, fpath: Path, top_k: int = TOP_K,
) -> None:
    top = summary.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4C72B0" if s > 0 else "#C44E52" for s in top["mean_signed_shap"]]
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_xlabel("mean |SHAP value| (impact on model output)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subject-level cluster bootstrap
# ---------------------------------------------------------------------------


def subject_bootstrap_logreg(
    X: np.ndarray, y: np.ndarray, subj: np.ndarray,
    feat_names: list[str], n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    unique_subj = np.unique(subj)
    subj_to_rows = {s: np.where(subj == s)[0] for s in unique_subj}

    coefs = np.full((n_boot, X.shape[1]), np.nan, dtype=np.float64)

    for b in tqdm(range(n_boot), desc="LogReg bootstrap"):
        sample_subj = rng.choice(unique_subj, size=len(unique_subj), replace=True)
        rows = np.concatenate([subj_to_rows[s] for s in sample_subj])
        Xb, yb = X[rows], y[rows]
        if len(np.unique(yb)) < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        C=0.1, max_iter=2000, class_weight="balanced",
                        random_state=random_state,
                    )),
                ])
                pipe.fit(Xb, yb)
                coefs[b] = pipe.named_steps["clf"].coef_[0]
        except Exception:
            continue

    valid = ~np.isnan(coefs).any(axis=1)
    coefs_ok = coefs[valid]

    summary = pd.DataFrame({
        "feature":   feat_names,
        "coef_mean": coefs_ok.mean(axis=0),
        "coef_lo":   np.percentile(coefs_ok, 2.5,  axis=0),
        "coef_hi":   np.percentile(coefs_ok, 97.5, axis=0),
    })
    summary["abs_mean"]         = summary["coef_mean"].abs()
    summary["ci_excludes_zero"] = (summary["coef_lo"] > 0) | (summary["coef_hi"] < 0)
    summary = summary.sort_values("abs_mean", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def plot_logreg_ci(
    summary: pd.DataFrame, title: str, fpath: Path, top_k: int = TOP_K,
) -> None:
    top = summary.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(top))
    err = np.array([
        top["coef_mean"].values - top["coef_lo"].values,
        top["coef_hi"].values  - top["coef_mean"].values,
    ])
    colors = ["#4C72B0" if c > 0 else "#C44E52" for c in top["coef_mean"]]
    ax.errorbar(top["coef_mean"], y_pos, xerr=err, fmt="none",
                ecolor="gray", capsize=3, zorder=1)
    for i, c in enumerate(colors):
        ax.scatter(top["coef_mean"].iloc[i], i, color=c, s=60, zorder=3)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("L2 LogReg coefficient (positive => increases P(PD))")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature provenance
# ---------------------------------------------------------------------------


def build_provenance(
    pooled_shap: pd.DataFrame, leak_shap: pd.DataFrame, logreg_ci: pd.DataFrame,
    stats_path: Path, top_k: int = TOP_K, leak_top_k: int = 20,
) -> pd.DataFrame:
    a_explore = pd.read_csv(stats_path)
    sig_per_feat = (
        a_explore[a_explore["q"] < 0.05]
        .groupby("feature").size().rename("stats_n_tasks_sig")
    )
    leak_rank = leak_shap.set_index("feature")["rank"].rename("leak_shap_rank")
    logreg = (
        logreg_ci.set_index("feature")[["coef_mean", "ci_excludes_zero", "rank"]]
        .rename(columns={"rank": "logreg_rank"})
    )

    top = (
        pooled_shap.head(top_k).set_index("feature")
        .rename(columns={"rank": "pooled_shap_rank"})
    )
    out = top.join(leak_rank).join(logreg).join(sig_per_feat)
    out["stats_n_tasks_sig"] = out["stats_n_tasks_sig"].fillna(0).astype(int)
    out[f"equipment_top{leak_top_k}"] = (
        out["leak_shap_rank"].fillna(np.inf) <= leak_top_k
    )
    return out.reset_index()


def plot_provenance_heatmap(
    prov: pd.DataFrame, fpath: Path, leak_top_k: int = 20,
) -> None:
    cols = pd.DataFrame({
        f"Pooled SHAP top-{len(prov)}":          np.ones(len(prov), dtype=int),
        f"Equipment SHAP top-{leak_top_k}":      prov[f"equipment_top{leak_top_k}"]
                                                    .astype(int).values,
        "LogReg 95% CI excludes 0":              prov["ci_excludes_zero"]
                                                    .fillna(False).astype(int).values,
        "Stats Section A sig (>=1 task)":        (prov["stats_n_tasks_sig"] >= 1)
                                                    .astype(int).values,
    }, index=prov["feature"]).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.32 * len(prov))))
    sns.heatmap(
        cols, cmap=["#FFFFFF", "#4C72B0"], cbar=False,
        linewidths=0.5, linecolor="lightgray", annot=False, ax=ax,
    )
    ax.set_title(
        f"Feature provenance — top-{len(prov)} pooled-SHAP features × evidence",
        fontsize=11, pad=12,
    )
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid")

    df, feat_cols = load_data()

    # Per-task: VA1 + RF + eGeMAPS
    sub = df[df["task_code"] == PER_TASK_TASK].copy()
    X_t = np.nan_to_num(sub[feat_cols].to_numpy(np.float64), nan=0.0)
    y_t = sub["y"].to_numpy()
    g_t = sub["subject_id"].to_numpy()

    rf_factory = lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    pt_summary, pt_sv, pt_xs = oof_shap(
        X_t, y_t, g_t, feat_cols, rf_factory, label=f"per-task {PER_TASK_TASK}",
    )
    pt_summary.to_csv(
        TABLES_DIR / f"per_task_{PER_TASK_TASK.lower()}_shap_top.csv", index=False,
    )
    plot_shap_bar(
        pt_summary,
        f"Top features — {PER_TASK_TASK} (RF, eGeMAPS, OOF SHAP)\n"
        "blue: ↑ feature ⇒ ↑ P(PD)   red: ↑ feature ⇒ ↓ P(PD)",
        FIG_DIR / f"per_task_{PER_TASK_TASK.lower()}_shap_bar.png",
    )
    plot_shap_summary(
        pt_sv, pt_xs, feat_cols,
        f"OOF SHAP — {PER_TASK_TASK} (RF, eGeMAPS)",
        FIG_DIR / f"per_task_{PER_TASK_TASK.lower()}_shap_summary.png",
    )

    # Pooled: GBM + eGeMAPS + task one-hot
    task_dum = pd.get_dummies(df["task_code"], prefix="task", dtype=float)
    X_p_feat = np.nan_to_num(df[feat_cols].to_numpy(np.float64), nan=0.0)
    X_p = np.hstack([X_p_feat, task_dum.to_numpy()])
    pooled_names = feat_cols + list(task_dum.columns)
    y_p = df["y"].to_numpy()
    g_p = df["subject_id"].to_numpy()

    gbm_factory = lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=RANDOM_STATE,
    )
    p_summary, p_sv, p_xs = oof_shap(
        X_p, y_p, g_p, pooled_names, gbm_factory, label="pooled",
    )
    p_summary.to_csv(TABLES_DIR / "pooled_shap_top.csv", index=False)
    plot_shap_bar(
        p_summary,
        "Top features — pooled (GBM, eGeMAPS+task, OOF SHAP)\n"
        "blue: ↑ feature ⇒ ↑ P(PD)   red: ↑ feature ⇒ ↓ P(PD)",
        FIG_DIR / "pooled_shap_bar.png",
    )
    plot_shap_summary(
        p_sv, p_xs, pooled_names,
        "OOF SHAP — pooled (GBM, eGeMAPS+task)",
        FIG_DIR / "pooled_shap_summary.png",
    )

    # Equipment-leak SHAP: within PD, predict sample_rate
    pd_only = df[df["y"] == 1].copy()
    pd_only["y_sr"] = (pd_only["sample_rate"] == 44100).astype(int)
    task_dum_l = pd.get_dummies(pd_only["task_code"], prefix="task", dtype=float)
    X_l_feat = np.nan_to_num(pd_only[feat_cols].to_numpy(np.float64), nan=0.0)
    X_l = np.hstack([X_l_feat, task_dum_l.to_numpy()])
    leak_names = feat_cols + list(task_dum_l.columns)
    y_l = pd_only["y_sr"].to_numpy()
    g_l = pd_only["subject_id"].to_numpy()

    l_summary, _, _ = oof_shap(X_l, y_l, g_l, leak_names, gbm_factory, label="leakage")
    l_summary.to_csv(TABLES_DIR / "leakage_shap_top.csv", index=False)
    plot_shap_bar(
        l_summary,
        "Top features — equipment leak (within PD, sample_rate, GBM, OOF SHAP)\n"
        "blue: ↑ feature ⇒ ↑ P(44.1 kHz)   red: ↑ feature ⇒ ↓ P(44.1 kHz)",
        FIG_DIR / "leakage_shap_bar.png",
    )

    # Subject-level bootstrap on L2 LogReg coefs
    lr_summary = subject_bootstrap_logreg(X_p, y_p, g_p, pooled_names)
    lr_summary.to_csv(TABLES_DIR / "pooled_logreg_coefs_ci.csv", index=False)
    plot_logreg_ci(
        lr_summary,
        "Top L2 LogReg coefs — pooled (eGeMAPS+task)\n"
        f"Subject cluster bootstrap, 95% CI from {N_BOOT} resamples",
        FIG_DIR / "pooled_logreg_coefs_ci.png",
    )

    # Feature provenance
    prov = build_provenance(
        p_summary, l_summary, lr_summary,
        stats_path=STATS_DIR / "exploratory_pd_vs_elderly.csv",
    )
    prov.to_csv(TABLES_DIR / "feature_provenance.csv", index=False)
    plot_provenance_heatmap(prov, FIG_DIR / "feature_provenance_heatmap.png")

    print("\nProvenance:")
    print(prov.to_string(index=False))

    print(f"\nTables: {TABLES_DIR.relative_to(PROJECT_ROOT)}/ | Figures: {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    run()
