from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
INTERIM = PROJECT_ROOT / "data" / "interim"
INTERIM_AUDIO = INTERIM / "audio"

TARGET_SR = 16_000

GROUP_FOLDERS = {
    "young_hc": "15 Young Healthy Control",
    "elderly_hc": "22 Elderly Healthy Control",
    "pd": "28 People with Parkinson's disease",
}

SUBJECT_FILES = {
    "young_hc": "15 YHC.xlsx",
    "elderly_hc": "Tab 3.xlsx",
    "pd": "TAB 5.xlsx",
}

FOLDER_ALIASES = {
    "liscog": "giuseppel",
    "porcellia": "antoniop",
    "summol": "luigias",
    "nicolòc": "nicolo'c",
    "mariob": "mariom",
    "antoniettap": "antonellap",
}

TASK_PATTERN = re.compile(r"^(FB\d|V[AEIOU]\d|[BD]\d|PR\d)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def normalize_key(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s))
    return re.sub(r"[\s\*]", "", s).lower()


def extract_date(filename: str) -> str | None:
    m = re.search(r"(\d{8})\d{4}\.+wav$", filename)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6})\d{4}\.+wav$", filename)
    if m:
        d = m.group(1)
        return d[:4] + "20" + d[4:]
    return None


def extract_task(filename: str) -> str:
    m = TASK_PATTERN.match(filename)
    return m.group(1).upper() if m else "UNKNOWN"


# ---------------------------------------------------------------------------
# Subjects table
# ---------------------------------------------------------------------------


def _read_one_subjects_file(path: Path, group: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=[0, 1])
    df.columns = [str(sub).strip().lower().replace(" ", "") for _, sub in df.columns]
    df = df[df["name"].notna() & (df["name"].astype(str).str.lower() != "name")].copy()
    df["group"] = group
    return df


def build_subjects_table() -> pd.DataFrame:
    frames = []
    for group, fname in SUBJECT_FILES.items():
        df = _read_one_subjects_file(RAW / GROUP_FOLDERS[group] / fname, group)
        frames.append(df)

    subjects = pd.concat(frames, ignore_index=True)

    if "unnamed:0_level_1" in subjects.columns:
        subjects = subjects.drop(columns=["unnamed:0_level_1"])

    subjects = subjects[subjects["age"].notna()].copy()

    subjects["folder_key"] = subjects.apply(
        lambda r: normalize_key(r["name"]) + normalize_key(r["surname"]), axis=1
    )

    subjects["subject_id"] = subjects["group"] + "_" + subjects["folder_key"]
    subjects = subjects.sort_values(["subject_id", "age"]).reset_index(drop=True)
    subjects["session"] = subjects.groupby("subject_id").cumcount() + 1
    subjects["subject_session_id"] = (
        subjects["subject_id"] + "_s" + subjects["session"].astype(str)
    )

    return subjects


# ---------------------------------------------------------------------------
# Audio file inventory
# ---------------------------------------------------------------------------


def _group_from_path(parts: tuple[str, ...]) -> str | None:
    top = parts[0]
    if "Young" in top:
        return "young_hc"
    if "Elderly" in top:
        return "elderly_hc"
    if "Parkinson" in top:
        return "pd"
    return None


def build_audio_manifest() -> pd.DataFrame:
    records = []
    for wav in RAW.rglob("*.wav"):
        rel = wav.relative_to(RAW)
        parts = rel.parts
        group = _group_from_path(parts)
        if group is None:
            continue
        records.append(
            {
                "file_path": str(rel),
                "group": group,
                "subject_folder": parts[-2],
                "task_code": extract_task(wav.name),
                "filename": wav.name,
                "date_str": extract_date(wav.name),
            }
        )

    manifest = pd.DataFrame(records)
    manifest["folder_key"] = manifest["subject_folder"].apply(normalize_key)
    manifest["folder_key"] = manifest["folder_key"].replace(FOLDER_ALIASES)
    manifest["date"] = pd.to_datetime(manifest["date_str"], format="%d%m%Y")

    session_map = (
        manifest[["group", "folder_key", "date"]]
        .drop_duplicates()
        .sort_values(["group", "folder_key", "date"])
    )
    session_map["session"] = session_map.groupby(["group", "folder_key"]).cumcount() + 1
    manifest = manifest.merge(session_map, on=["group", "folder_key", "date"])

    return manifest


# ---------------------------------------------------------------------------
# Audio resampling and QC
# ---------------------------------------------------------------------------


def resample_and_qc(manifest: pd.DataFrame) -> pd.DataFrame:
    INTERIM_AUDIO.mkdir(parents=True, exist_ok=True)

    qc_records = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Resampling"):
        src = RAW / row["file_path"]
        out_dir = INTERIM_AUDIO / row["group"] / row["subject_session_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / row["filename"]

        y, _ = librosa.load(src, sr=TARGET_SR, mono=True)

        rms = float(np.sqrt(np.mean(y**2)))
        peak = float(np.max(np.abs(y)))

        sf.write(dst, y, TARGET_SR, subtype="PCM_16")

        qc_records.append(
            {
                "file_path": row["file_path"],
                "interim_path": str(dst.relative_to(INTERIM)),
                "rms": rms,
                "peak": peak,
                "duration_sec_resampled": len(y) / TARGET_SR,
            }
        )

    qc = pd.DataFrame(qc_records)
    qc["flag_clipped"] = qc["peak"] >= 0.999
    qc["flag_too_quiet"] = qc["rms"] < 0.005
    qc["flag_too_short"] = qc["duration_sec_resampled"] < 1.0
    qc["qc_pass"] = ~(qc["flag_clipped"] | qc["flag_too_quiet"] | qc["flag_too_short"])
    return qc


# ---------------------------------------------------------------------------
# Source audio properties
# ---------------------------------------------------------------------------


def probe_source_audio(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Probing"):
        info = sf.info(RAW / row["file_path"])
        rows.append(
            {
                "file_path": row["file_path"],
                "sample_rate": info.samplerate,
                "n_channels": info.channels,
                "duration_sec": info.duration,
                "n_frames": info.frames,
                "subtype": info.subtype,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)

    subjects = build_subjects_table()
    # laforgia v has no audio folder
    subjects = subjects[subjects["folder_key"] != "laforgiav"].copy()
    subjects.to_csv(INTERIM / "subjects.csv", index=False)

    manifest = build_audio_manifest()

    manifest_full = manifest.merge(
        subjects[
            [
                "group",
                "folder_key",
                "session",
                "subject_id",
                "subject_session_id",
                "sex",
                "age",
                "cps1",
                "cps2",
                "cps3",
            ]
        ],
        on=["group", "folder_key", "session"],
        how="left",
    )
    manifest_full = manifest_full.dropna(subset=["subject_id"]).copy()

    src_info = probe_source_audio(manifest_full)
    manifest_full = manifest_full.merge(src_info, on="file_path")

    qc = resample_and_qc(manifest_full)
    manifest_full = manifest_full.merge(qc, on="file_path")

    manifest_full = manifest_full.drop(columns=["folder_key"])
    manifest_full.to_csv(INTERIM / "manifest.csv", index=False)

    print(f"Total files: {len(manifest_full)} | Pass QC: {manifest_full['qc_pass'].sum()}")
    print(f"Outputs: {(INTERIM / 'subjects.csv').relative_to(PROJECT_ROOT)}, "
          f"{(INTERIM / 'manifest.csv').relative_to(PROJECT_ROOT)}, "
          f"{INTERIM_AUDIO.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    run()
