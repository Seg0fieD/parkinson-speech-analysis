from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "stats"

COMPARABLE_TASKS = ["B1", "B2", "PR1"]

PRIMARY_MARKERS = {
    "jitter":     "jitterLocal_sma3nz_amean",
    "shimmer":    "shimmerLocaldB_sma3nz_amean",
    "hnr":        "HNRdBACF_sma3nz_amean",
    "f0_mean":    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "f0_var":     "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness":   "loudness_sma3_amean",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(INTERIM / "manifest.csv")
    manifest = manifest[manifest["qc_pass"] & (manifest["session"] == 1)].copy()

    egemaps = pd.read_parquet(PROCESSED / "features_egemaps.parquet")
    df = manifest.merge(egemaps, on="file_path", validate="1:1")

    feat_cols = [c for c in egemaps.columns if c != "file_path"]
    return df, feat_cols


# ---------------------------------------------------------------------------
# Per-task OLS fitters
# ---------------------------------------------------------------------------


def _fit_pd_vs_elderly(
    df: pd.DataFrame, task: str, feature: str, marker_label: str,
    extra_covariates: list[str] | None = None,
) -> dict:
    extra_covariates = extra_covariates or []
    cols = [feature, "group", "age", "sex"] + extra_covariates
    sub = df[(df["task_code"] == task) & (df["group"].isin(["pd", "elderly_hc"]))]
    sub = sub[cols].dropna()
    n_pd = (sub["group"] == "pd").sum()
    n_elderly = (sub["group"] == "elderly_hc").sum()
    base = {
        "task": task, "marker": marker_label, "feature": feature,
        "n": len(sub), "n_pd": int(n_pd), "n_elderly": int(n_elderly),
    }
    if n_pd < 3 or n_elderly < 3:
        return {**base, "beta_pd": np.nan, "se": np.nan, "t": np.nan, "p": np.nan}

    extra_terms = "".join(f" + {c}" for c in extra_covariates)
    formula = (
        f'Q("{feature}") ~ C(group, Treatment("elderly_hc")) + age + C(sex)'
        f"{extra_terms}"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=sub).fit()
    coef_name = 'C(group, Treatment("elderly_hc"))[T.pd]'
    return {
        **base,
        "beta_pd":  float(model.params.get(coef_name, np.nan)),
        "se":       float(model.bse.get(coef_name, np.nan)),
        "t":        float(model.tvalues.get(coef_name, np.nan)),
        "p":        float(model.pvalues.get(coef_name, np.nan)),
    }


def _fit_3group(
    df: pd.DataFrame, task: str, feature: str, marker_label: str
) -> dict:
    sub = df[df["task_code"] == task][[feature, "group", "age", "sex"]].dropna()
    counts = sub["group"].value_counts()
    base = {
        "task": task, "marker": marker_label, "feature": feature,
        "n": len(sub),
        "n_young":   int(counts.get("young_hc", 0)),
        "n_elderly": int(counts.get("elderly_hc", 0)),
        "n_pd":      int(counts.get("pd", 0)),
    }
    if min(counts.get(g, 0) for g in ["young_hc", "elderly_hc", "pd"]) < 3:
        return {
            **base, "F_group": np.nan, "p_group": np.nan,
            "beta_pd": np.nan, "p_pd": np.nan,
            "beta_young": np.nan, "p_young": np.nan,
        }

    formula = (
        f'Q("{feature}") ~ C(group, Treatment("elderly_hc")) + age + C(sex)'
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=sub).fit()

    # Joint F-test for group effect
    group_terms = [
        'C(group, Treatment("elderly_hc"))[T.pd]',
        'C(group, Treatment("elderly_hc"))[T.young_hc]',
    ]
    group_terms = [t for t in group_terms if t in model.params.index]
    f_test = model.f_test(" = 0, ".join(group_terms) + " = 0") if group_terms else None

    return {
        **base,
        "F_group":    float(f_test.fvalue) if f_test is not None else np.nan,
        "p_group":    float(f_test.pvalue) if f_test is not None else np.nan,
        "beta_pd":    float(model.params.get(group_terms[0], np.nan)) if group_terms else np.nan,
        "p_pd":       float(model.pvalues.get(group_terms[0], np.nan)) if group_terms else np.nan,
        "beta_young": float(model.params.get(group_terms[1], np.nan)) if len(group_terms) > 1 else np.nan,
        "p_young":    float(model.pvalues.get(group_terms[1], np.nan)) if len(group_terms) > 1 else np.nan,
    }


# ---------------------------------------------------------------------------
# Family runners
# ---------------------------------------------------------------------------


def _add_fdr(rows: pd.DataFrame, p_col: str, q_col: str) -> pd.DataFrame:
    rows = rows.copy()
    mask = rows[p_col].notna()
    q = np.full(len(rows), np.nan)
    if mask.any():
        _, q_valid, _, _ = multipletests(rows.loc[mask, p_col].values, method="fdr_bh")
        q[mask.values] = q_valid
    rows[q_col] = q
    return rows


def section_a(df: pd.DataFrame, features: dict[str, str], desc: str) -> pd.DataFrame:
    tasks = sorted(df["task_code"].unique())
    rows = []
    iterator = ((t, lab, feat) for t in tasks for lab, feat in features.items())
    total = len(tasks) * len(features)
    for t, lab, feat in tqdm(iterator, total=total, desc=desc):
        rows.append(_fit_pd_vs_elderly(df, t, feat, lab))
    return _add_fdr(pd.DataFrame(rows), "p", "q")


def section_b(df: pd.DataFrame, features: dict[str, str], desc: str) -> pd.DataFrame:
    rows = []
    iterator = ((t, lab, feat) for t in COMPARABLE_TASKS for lab, feat in features.items())
    total = len(COMPARABLE_TASKS) * len(features)
    for t, lab, feat in tqdm(iterator, total=total, desc=desc):
        rows.append(_fit_3group(df, t, feat, lab))
    out = pd.DataFrame(rows)
    out = _add_fdr(out, "p_group", "q_group")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _summarize_section_a(name: str, df: pd.DataFrame) -> None:
    sig = df[df["q"] < 0.05].sort_values("q")
    print(f"\n[{name}] {len(sig)}/{len(df)} significant at q<0.05")
    if len(sig):
        cols = ["task", "marker", "n", "beta_pd", "t", "p", "q"]
        cols = [c for c in cols if c in sig.columns]
        print(sig[cols].head(15).to_string(index=False))


def _summarize_section_b(name: str, df: pd.DataFrame) -> None:
    sig = df[df["q_group"] < 0.05].sort_values("q_group")
    print(f"\n[{name}] {len(sig)}/{len(df)} significant at q<0.05 (group F-test)")
    if len(sig):
        cols = ["task", "marker", "n", "F_group", "p_group", "q_group",
                "beta_pd", "p_pd", "beta_young", "p_young"]
        cols = [c for c in cols if c in sig.columns]
        print(sig[cols].head(15).to_string(index=False))


# ---------------------------------------------------------------------------
# Sensitivity checks
# ---------------------------------------------------------------------------


def _section_a_variant(
    df: pd.DataFrame, features: dict[str, str], desc: str,
    extra_covariates: list[str] | None = None,
) -> pd.DataFrame:
    tasks = sorted(df["task_code"].unique())
    rows = []
    iterator = ((t, lab, feat) for t in tasks for lab, feat in features.items())
    total = len(tasks) * len(features)
    for t, lab, feat in tqdm(iterator, total=total, desc=desc):
        rows.append(
            _fit_pd_vs_elderly(df, t, feat, lab, extra_covariates=extra_covariates)
        )
    return _add_fdr(pd.DataFrame(rows), "p", "q")


def sensitivity_checks(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    # S1: + sample_rate covariate
    s1 = _section_a_variant(
        df, PRIMARY_MARKERS, "S1: +sample_rate", extra_covariates=["sample_rate"]
    )

    # S2: stricter QC
    df_strict = df[(df["peak"] < 0.95) & (df["rms"] > 0.01)].copy()
    s2 = _section_a_variant(df_strict, PRIMARY_MARKERS, "S2: strict QC")

    keep = ["task", "marker", "n", "beta_pd", "p", "q"]
    cmp = (
        baseline[keep].rename(columns={c: f"{c}_base" for c in keep[2:]})
        .merge(
            s1[keep].rename(columns={c: f"{c}_s1" for c in keep[2:]}),
            on=["task", "marker"],
        )
        .merge(
            s2[keep].rename(columns={c: f"{c}_s2" for c in keep[2:]}),
            on=["task", "marker"],
        )
    )
    return cmp


def _summarize_sensitivity(cmp: pd.DataFrame) -> None:
    base_sig = cmp[cmp["q_base"] < 0.05].copy()
    base_sig["s1_flipped"] = base_sig["q_s1"] >= 0.05
    base_sig["s2_flipped"] = base_sig["q_s2"] >= 0.05

    n_base = len(base_sig)
    n_s1_flip = int(base_sig["s1_flipped"].sum())
    n_s2_flip = int(base_sig["s2_flipped"].sum())
    print(f"\nBaseline significant: {n_base} | Lost under S1: {n_s1_flip} | Lost under S2: {n_s2_flip}")

    flipped = base_sig[base_sig["s1_flipped"] | base_sig["s2_flipped"]]
    if len(flipped):
        cols = ["task", "marker", "beta_pd_base", "q_base",
                "beta_pd_s1", "q_s1", "beta_pd_s2", "q_s2"]
        print(flipped[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df, feat_cols = load_data()
    all_features = {c: c for c in feat_cols}

    a_primary = section_a(df, PRIMARY_MARKERS, "A primary")
    a_primary.to_csv(TABLES_DIR / "primary_pd_vs_elderly.csv", index=False)
    _summarize_section_a("primary PD vs elderly", a_primary)

    b_primary = section_b(df, PRIMARY_MARKERS, "B primary")
    b_primary.to_csv(TABLES_DIR / "primary_3group_b1b2pr1.csv", index=False)
    _summarize_section_b("primary 3-group", b_primary)

    sens = sensitivity_checks(df, a_primary)
    sens.to_csv(TABLES_DIR / "sensitivity_pd_vs_elderly.csv", index=False)
    _summarize_sensitivity(sens)

    a_explore = section_a(df, all_features, "A exploratory")
    a_explore.to_csv(TABLES_DIR / "exploratory_pd_vs_elderly.csv", index=False)
    _summarize_section_a("exploratory PD vs elderly", a_explore)

    b_explore = section_b(df, all_features, "B exploratory")
    b_explore.to_csv(TABLES_DIR / "exploratory_3group_b1b2pr1.csv", index=False)
    _summarize_section_b("exploratory 3-group", b_explore)

    print(f"\nTables: {TABLES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    run()
