from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
TARGET_SR = 16_000
HIDDEN_DIM = 1024

CHUNK_SEC = 30.0
OVERLAP_SEC = 5.0
CHUNK_LEN = int(CHUNK_SEC * TARGET_SR)
HOP_LEN = int((CHUNK_SEC - OVERLAP_SEC) * TARGET_SR)

CHECKPOINT_EVERY = 50
OUT_PATH = PROCESSED / "features_w2v2.parquet"


# ---------------------------------------------------------------------------
# Device + model
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(device: torch.device) -> tuple[Wav2Vec2FeatureExtractor, Wav2Vec2Model]:
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    return fe, model


# ---------------------------------------------------------------------------
# Per-chunk + per-file embedding
# ---------------------------------------------------------------------------


def _chunk_audio(y: np.ndarray) -> list[np.ndarray]:
    if len(y) <= CHUNK_LEN:
        return [y]
    chunks = []
    start = 0
    while start < len(y):
        end = start + CHUNK_LEN
        chunks.append(y[start:end])
        if end >= len(y):
            break
        start += HOP_LEN
    return chunks


@torch.no_grad()
def _embed_chunk(
    chunk: np.ndarray,
    fe: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> np.ndarray:
    inputs = fe(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    out = model(input_values)
    h = out.last_hidden_state.squeeze(0)
    mean = h.mean(dim=0)
    std = h.std(dim=0)
    return torch.cat([mean, std], dim=0).cpu().numpy().astype(np.float32)


def _embed_file(
    audio_path: Path,
    fe: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> np.ndarray:
    y, sr = sf.read(str(audio_path), dtype="float32")
    assert sr == TARGET_SR, f"Expected {TARGET_SR} Hz, got {sr} Hz at {audio_path}"
    if y.ndim > 1:
        y = y.mean(axis=1)

    chunks = _chunk_audio(y)
    chunk_vecs = [_embed_chunk(c, fe, model, device) for c in chunks]
    return np.mean(chunk_vecs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------


def _feature_columns() -> list[str]:
    return [f"w2v2_mean_{i:04d}" for i in range(HIDDEN_DIM)] + [
        f"w2v2_std_{i:04d}" for i in range(HIDDEN_DIM)
    ]


def _load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        return pd.read_parquet(OUT_PATH)
    return pd.DataFrame(columns=["file_path"] + _feature_columns())


def _save_checkpoint(rows: list[dict], existing: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return existing
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_parquet(OUT_PATH, index=False)
    return combined


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_w2v2(manifest: pd.DataFrame) -> pd.DataFrame:
    device = _select_device()

    existing = _load_existing()
    done_paths = set(existing["file_path"].tolist())
    todo = manifest[~manifest["file_path"].isin(done_paths)].reset_index(drop=True)

    if len(todo) == 0:
        return existing

    fe, model = _load_model(device)
    feat_cols = _feature_columns()
    pending: list[dict] = []

    for i, (_, row) in enumerate(
        tqdm(todo.iterrows(), total=len(todo), desc=f"wav2vec2 [{device}]"), start=1
    ):
        audio_path = INTERIM / row["interim_path"]
        vec = _embed_file(audio_path, fe, model, device)
        pending.append({"file_path": row["file_path"], **dict(zip(feat_cols, vec))})

        if i % CHECKPOINT_EVERY == 0:
            existing = _save_checkpoint(pending, existing)
            pending = []

    existing = _save_checkpoint(pending, existing)
    return existing


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(INTERIM / "manifest.csv")
    manifest = manifest[manifest["qc_pass"]].reset_index(drop=True)

    feats = extract_w2v2(manifest)

    print(f"Files: {len(feats)} | Features: {feats.shape[1] - 1} | Output: {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
