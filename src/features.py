from __future__ import annotations

from pathlib import Path

import opensmile
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# eGeMAPS extraction
# ---------------------------------------------------------------------------


def _make_egemaps_extractor() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps(manifest: pd.DataFrame) -> pd.DataFrame:
    smile = _make_egemaps_extractor()

    frames = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="eGeMAPS"):
        audio_path = INTERIM / row["interim_path"]
        feats = smile.process_file(str(audio_path))
        feats = feats.reset_index(drop=True)
        feats.insert(0, "file_path", row["file_path"])
        frames.append(feats)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(INTERIM / "manifest.csv")
    manifest = manifest[manifest["qc_pass"]].reset_index(drop=True)

    feats = extract_egemaps(manifest)
    out_path = PROCESSED / "features_egemaps.parquet"
    feats.to_parquet(out_path, index=False)

    print(f"Files: {len(feats)} | Features: {feats.shape[1] - 1} | Output: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
