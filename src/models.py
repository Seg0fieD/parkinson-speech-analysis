from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "models"

N_FOLDS = 5
RANDOM_STATE = 0
MIN_SUBJECTS_PER_CLASS = N_FOLDS


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def get_classifiers() -> dict[str, object]:
    clfs = {
        "logreg": LogisticRegression(
            C=0.1, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gbm": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=RANDOM_STATE,
        ),
    }
    if HAS_XGB:
        clfs["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
            verbosity=0,
        )
    return clfs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    manifest = pd.read_csv(INTERIM / "manifest.csv")
    manifest = manifest[
        manifest["qc_pass"]
        & (manifest["session"] == 1)
        & (manifest["group"].isin(["pd", "elderly_hc"]))
    ].copy()
    manifest["y"] = (manifest["group"] == "pd").astype(int)

    egemaps = pd.read_parquet(PROCESSED / "features_egemaps.parquet")
    w2v2 = pd.read_parquet(PROCESSED / "features_w2v2.parquet")

    df = manifest.merge(egemaps, on="file_path", validate="1:1")
    df = df.merge(w2v2, on="file_path", validate="1:1")

    ege_cols = [c for c in egemaps.columns if c != "file_path"]
    wv_cols = [c for c in w2v2.columns if c != "file_path"]

    return df, ege_cols, wv_cols


# ---------------------------------------------------------------------------
# Cross-validation core
# ---------------------------------------------------------------------------


def cv_evaluate(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, clf, n_folds: int = N_FOLDS,
) -> list[dict]:
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr], y[tr])
            proba = pipe.predict_proba(X[te])[:, 1]
            pred = pipe.predict(X[te])
        rows.append({
            "fold": fold,
            "n_train": len(tr), "n_test": len(te),
            "auc":     roc_auc_score(y[te], proba) if len(np.unique(y[te])) > 1 else np.nan,
            "bal_acc": balanced_accuracy_score(y[te], pred),
            "f1":      f1_score(y[te], pred, zero_division=0),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------


def _has_enough(df: pd.DataFrame) -> bool:
    sub = df.drop_duplicates("subject_id")
    counts = sub["y"].value_counts()
    return counts.get(0, 0) >= MIN_SUBJECTS_PER_CLASS and counts.get(1, 0) >= MIN_SUBJECTS_PER_CLASS


def per_task_evaluation(
    df: pd.DataFrame, ege_cols: list[str], wv_cols: list[str],
) -> pd.DataFrame:
    classifiers = get_classifiers()
    feature_sets = {"egemaps": ege_cols, "w2v2": wv_cols}
    tasks = sorted(df["task_code"].unique())

    total = len(tasks) * len(classifiers) * len(feature_sets)
    pbar = tqdm(total=total, desc="per-task CV")
    rows = []

    for task in tasks:
        sub = df[df["task_code"] == task]
        if not _has_enough(sub):
            pbar.update(len(classifiers) * len(feature_sets))
            continue
        groups = sub["subject_id"].to_numpy()
        y = sub["y"].to_numpy()
        for fs_name, cols in feature_sets.items():
            X = sub[cols].to_numpy(dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0)
            for clf_name, clf in classifiers.items():
                fold_rows = cv_evaluate(X, y, groups, clf)
                for r in fold_rows:
                    rows.append({
                        "task": task, "classifier": clf_name,
                        "feature_set": fs_name, **r,
                    })
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled evaluation
# ---------------------------------------------------------------------------


def pooled_evaluation(
    df: pd.DataFrame, ege_cols: list[str], wv_cols: list[str],
) -> pd.DataFrame:
    classifiers = get_classifiers()
    feature_sets = {"egemaps": ege_cols, "w2v2": wv_cols}
    task_dummies = pd.get_dummies(df["task_code"], prefix="task", dtype=float)

    groups = df["subject_id"].to_numpy()
    y = df["y"].to_numpy()

    rows = []
    pbar = tqdm(total=len(classifiers) * len(feature_sets), desc="pooled CV")
    for fs_name, cols in feature_sets.items():
        X_feat = df[cols].to_numpy(dtype=np.float64)
        X_feat = np.nan_to_num(X_feat, nan=0.0)
        X = np.hstack([X_feat, task_dummies.to_numpy()])
        for clf_name, clf in classifiers.items():
            fold_rows = cv_evaluate(X, y, groups, clf)
            for r in fold_rows:
                rows.append({"classifier": clf_name, "feature_set": fs_name, **r})
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leakage check: within-PD sample-rate prediction
# ---------------------------------------------------------------------------


def leakage_sample_rate_check(
    df: pd.DataFrame, ege_cols: list[str], wv_cols: list[str],
) -> pd.DataFrame:
    pd_only = df[df["y"] == 1].copy()
    pd_only["y_sr"] = (pd_only["sample_rate"] == 44100).astype(int)

    sub_class = pd_only.drop_duplicates("subject_id")["y_sr"].value_counts()
    if sub_class.min() < MIN_SUBJECTS_PER_CLASS:
        return pd.DataFrame()

    classifiers = get_classifiers()
    feature_sets = {"egemaps": ege_cols, "w2v2": wv_cols}
    task_dummies = pd.get_dummies(pd_only["task_code"], prefix="task", dtype=float)
    groups = pd_only["subject_id"].to_numpy()
    y = pd_only["y_sr"].to_numpy()

    rows = []
    pbar = tqdm(total=len(classifiers) * len(feature_sets), desc="leakage CV")
    for fs_name, cols in feature_sets.items():
        X_feat = pd_only[cols].to_numpy(dtype=np.float64)
        X_feat = np.nan_to_num(X_feat, nan=0.0)
        X = np.hstack([X_feat, task_dummies.to_numpy()])
        for clf_name, clf in classifiers.items():
            fold_rows = cv_evaluate(X, y, groups, clf)
            for r in fold_rows:
                rows.append({"classifier": clf_name, "feature_set": fs_name, **r})
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def summarize(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    agg = results.groupby(group_cols).agg(
        n_folds=("fold", "count"),
        auc_mean=("auc", "mean"),     auc_std=("auc", "std"),
        bal_acc_mean=("bal_acc", "mean"), bal_acc_std=("bal_acc", "std"),
        f1_mean=("f1", "mean"),       f1_std=("f1", "std"),
    ).round(3).reset_index()
    return agg


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df, ege_cols, wv_cols = load_data()

    per_task = per_task_evaluation(df, ege_cols, wv_cols)
    per_task.to_csv(TABLES_DIR / "per_task_cv_results.csv", index=False)
    per_task_sum = summarize(per_task, ["task", "feature_set", "classifier"])
    per_task_sum.to_csv(TABLES_DIR / "per_task_summary.csv", index=False)

    print("\nTop 10 per-task by AUC:")
    print(
        per_task_sum.sort_values("auc_mean", ascending=False)
        .head(10).to_string(index=False)
    )

    pooled = pooled_evaluation(df, ege_cols, wv_cols)
    pooled.to_csv(TABLES_DIR / "pooled_cv_results.csv", index=False)
    pooled_sum = summarize(pooled, ["feature_set", "classifier"])
    pooled_sum.to_csv(TABLES_DIR / "pooled_summary.csv", index=False)

    print("\nPooled results:")
    print(pooled_sum.to_string(index=False))

    leakage = leakage_sample_rate_check(df, ege_cols, wv_cols)
    if len(leakage):
        leakage.to_csv(TABLES_DIR / "leakage_cv_results.csv", index=False)
        leakage_sum = summarize(leakage, ["feature_set", "classifier"])
        leakage_sum.to_csv(TABLES_DIR / "leakage_summary.csv", index=False)
        print("\nEquipment-prediction AUC within PD:")
        print(leakage_sum.to_string(index=False))

    print(f"\nTables: {TABLES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    run()
