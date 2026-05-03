# Parkinson's Speech Analysis

End-to-end pipeline analyzing speech in Parkinson's disease using the [Italian Parkinson's Voice and Speech dataset](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech). Cross-sectional statistics, predictive modeling, longitudinal repeated-measures analysis, and interpretability — with explicit leakage and confound checks throughout.

This is a methodological project as much as a clinical one: the headline finding isn't an AUC, it's a recording-equipment confound that a naive pipeline would have missed and reported as state-of-the-art PD detection.

---

## TL;DR

- **Cross-sectional PD vs age-matched controls reaches AUC 0.99 on eGeMAPS features — but ~30% of that signal is recording-equipment, not pathology.** All 44.1 kHz files in the dataset are PD subjects; all elderly controls are 16 kHz. A within-PD sample-rate classifier hits AUC 0.92 from the same features, exposing the leak.
- **Real PD signal does survive.** 27 of 32 baseline FDR-significant findings remain after sensitivity checks (sample-rate covariate, strict QC). Reduced F0 variability (`stddevNorm`) and altered spectral slope (`slopeV0-500`) replicate across 16/16 tasks and through the cleanest interpretability layers.
- **Longitudinal inference is impossible from this dataset.** All three multi-session PD subjects switched recording equipment between session 1 and session 2 — equipment change and time effect are perfectly colinear. Documented and shipped as a negative result rather than reporting confounded coefficients.
- **eGeMAPS (88 hand-engineered features) outperforms wav2vec2 (2048 self-supervised dims) at this sample size** (45 subjects), per-task AUC 0.97 vs 0.95 for the best linear model. Higher dimensions also leaked more equipment signal.

---

## Dataset

| Group | Subjects | Files (qc_pass) | Tasks | Notes |
|---|---|---|---|---|
| `young_hc` | 15 | 45 | B1, B2, PR1 | All 16 kHz |
| `elderly_hc` | 21 | 312 | full 16-task set | All 16 kHz |
| `pd` | 28 | 433 | full 16-task set | 156 files at 44.1 kHz, 213 at 16 kHz; 3 subjects multi-session |

**Tasks:** B1/B2 (balanced texts), D1/D2 (/pa/, /ta/ syllables), FB1 (phrases), PR1 (passage), VA1–VU2 (sustained vowels).

**Final analysis cohort.** All findings condition on `qc_pass` (peak < 0.999, RMS ≥ 0.005, duration ≥ 1s) AND `session == 1`. 681 files, 45 subjects (24 PD + 21 elderly_hc, age-matched at ~67 years).

---

## Pipeline overview

```
preprocess.py     ─→ subjects.csv, manifest.csv, 16 kHz mono audio
features.py       ─→ eGeMAPSv02 functionals (88-d, openSMILE)
features_w2v2.py  ─→ wav2vec2 XLSR-53 mean+std embeddings (2048-d)
eda.py            ─→ cohort overview, PCA, UMAP, dysphonia markers
stats.py          ─→ per-task OLS, BH-FDR, sensitivity checks
models.py         ─→ 4 classifiers × 2 feature sets, StratifiedGroupKFold,
                     plus within-PD equipment-leak diagnostic
longitudinal.py   ─→ MixedLM on multi-session subjects + confound check
interpret.py      ─→ OOF SHAP, subject-cluster bootstrap, feature provenance
```

Cross-validation always uses `StratifiedGroupKFold` on `subject_id` — no subject ever appears in both train and test.

---

## Findings

### 1. Cross-sectional statistics (`stats.py`)

Per-task OLS for each acoustic marker, BH-FDR within family, session==1 only.

**Section A — PD vs elderly_hc, 16 tasks × 6 primary markers (96 tests):** 32/96 significant at q < 0.05.

Direction-of-effect on reading tasks (B1, B2, FB1, PR1):
- HNR ↑ in PD by ~2.0–2.2 dB 
- Shimmer ↓ in PD 
- F0 variability (`stddevNorm`) ↓ in PD *(consistent with monotone-speech literature)*
- Loudness ↑ in PD on sustained vowels

The "opposite of textbook" patterns aren't errors. The Section B 3-group analysis on the comparable tasks (B1/B2/PR1, n_young=15, n_elderly=18, n_pd=21) shows that elderly_hc is the high-shimmer / low-HNR group; PD looks similar to young_hc on these markers. Headline: **age-matched elderly controls in this dataset have voice changes (likely subclinical presbyphonia) that are larger than what mild-moderate PD adds on top.**

**Sensitivity checks.** Two variants: (S1) add `sample_rate` as a covariate, (S2) strict QC (peak < 0.95, RMS > 0.01).

| | Baseline q<0.05 | Lost under S1 | Lost under S2 |
|---|---|---|---|
| Primary markers | 32 | 5 | 1 |

**27/32 baseline findings (84%) survive both checks.** What was lost is informative: FB1/PR1 loudness and several sustained-vowel marginal results — exactly the spectral-energy-flavored findings that recording chain affects most.

**Section B — 3-group on B1/B2/PR1, 18 tests:** 11/18 significant.

**Exploratory — all 88 eGeMAPS features:** 430/1408 PD-vs-elderly tests significant. Top hits: `slopeUV0-500`, `slopeV0-500`, Hammarberg index, alpha ratio. Read in light of the leakage analysis below.

### 2. Predictive modeling (`models.py`)

PD vs elderly_hc, session 1 only, n=45 subjects, n=681 files. 4 classifiers × 2 feature sets × 5-fold StratifiedGroupKFold.

**Per-task AUC averaged across 16 tasks:**

| Feature set | LogReg | RF | GBM | XGBoost |
|---|---|---|---|---|
| eGeMAPS | 0.969 | **0.986** | 0.951 | 0.958 |
| wav2vec2 | 0.948 | 0.915 | 0.854 | 0.877 |

**Pooled (all tasks, task as one-hot):**

| Feature set | Best classifier | AUC ± std |
|---|---|---|
| eGeMAPS | GBM | **0.992 ± 0.010** |
| wav2vec2 | LogReg | 0.962 ± 0.057 |

**These numbers are partly equipment, not pathology.** The leakage diagnostic asks: within PD only, can you predict whether a recording was originally 16 kHz or 44.1 kHz?

| Feature set | Best classifier | AUC for sample-rate prediction within PD |
|---|---|---|
| eGeMAPS | GBM | **0.918** |
| wav2vec2 | LogReg | 0.78 |

A clean (no-leak) feature set should give chance performance (~0.5) here. 0.92 means eGeMAPS strongly encodes recording equipment. Since elderly_hc is 100% 16 kHz and PD is 42% 44.1 kHz, the PD-vs-elderly model can read equipment as a proxy for class. The 0.992 pooled AUC is best read as an upper bound, not the deployable performance.

eGeMAPS leaks more than wav2vec2 — interesting given their relative interpretability and curation. Likely because the spectral-shape functionals (slopes, alpha ratio, Hammarberg index) are precisely what changes when you resample 44.1 → 16 kHz audio.

### 3. Longitudinal — a deliberate negative result (`longitudinal.py`)

3 PD subjects have multiple recording sessions: Vito S (3), Roberto R (2), Nicola S (2).

A naive MixedLM (`marker ~ months_from_baseline + (1|subject)`) shows apparent worsening: HNR ↓ (p=1.5e-5), F0 mean ↓ (p=1.4e-5), loudness ↑ (p=6.9e-10).

**The confound check kills it.** All 3 subjects switched from 44.1 kHz at session 1 to 16 kHz at session 2. The equipment change perfectly coincides with the time variable, so the "time effect" is unidentifiable.

The trajectory plot (`reports/figures/longitudinal/trajectories.png`) encodes sample rate as marker shape so the equipment switch is visible. The output tables carry an `equipment_confounded=True` flag. **Longitudinal inference from this dataset is not valid, and that is the finding.**

### 4. Interpretability (`interpret.py`)

Three SHAP analyses (out-of-fold, refit per fold, no leakage) plus a subject-cluster bootstrap on L2 LogReg coefficients.

**Pooled GBM, top SHAP features:** `slopeUV0-500` (#1, mean |SHAP| 6.27 — 2.5× the next), `slopeV0-500`, `slopeUV500-1500`, `hammarbergIndexUV`, `loudness_sma3_stddevNorm`.

**Equipment-leak SHAP (within PD, predict sample_rate):** `slopeUV0-500` is also #1 with mean |SHAP| 4.35. The dominant feature in the PD model is the dominant feature in the equipment model.

**Feature provenance** cross-references each top-SHAP feature against (a) equipment-leak ranking, (b) bootstrapped LogReg CI, (c) Section A FDR-significant tasks count:

| Of top-15 pooled-SHAP features | Count |
|---|---|
| Likely real (NOT in equipment top-20 AND sig in ≥1 task) | 8 |
| Equipment-suspect (in within-PD sample-rate top-20) | 7 |
| LogReg 95% CI excludes 0 | 11 |

The cleanest "real" features — top SHAP, sig in ≥10 tasks, NOT in equipment top-20, LogReg CI excludes zero:

| Feature | What it is |
|---|---|
| `slopeV0-500_sma3nz_amean` | Spectral slope 0–500 Hz, voiced segments |
| `hammarbergIndexUV_sma3nz_amean` | Spectral peak ratio, unvoiced segments |
| `loudnessPeaksPerSec` | Rate of loudness peaks |
| `alphaRatioUV_sma3nz_amean` | High-vs-low frequency energy ratio, unvoiced |

These are spectral-envelope and prosodic features, consistent with established PD voice findings (hypophonia, monotone speech, reduced spectral tilt range).

`shimmerLocaldB`, the textbook dysphonia marker, is #8 in PD-vs-elderly SHAP but also #10 in the equipment model — methodologically interesting and in line with the audio literature on amplitude-perturbation features being recording-chain-sensitive.

**VA1 sustained /a/ alone:** None of jitter, HNR, or F0 variability appear in the top-10. The RF model uses spectral slope and flux variants instead. This is a clean answer to "what does the model use when classical primary markers don't separate?" — spectral envelope shape, not periodicity.

---

## Caveats

- **Recording-equipment confound is the dominant methodological issue.** The pipeline does not "fix" it (you can't fix a structural confound post-hoc); it documents it and reports findings stratified by how robust they are.
- **n = 45 subjects** for the PD-vs-elderly modeling. AUCs are high but variance estimates are correspondingly wide.
- **Cohort selection.** Elderly controls in this dataset have measurable voice changes (likely subclinical presbyphonia). PD vs younger or vocally-screened controls would likely show different — almost certainly larger — effect sizes for classical dysphonia markers.
- **Longitudinal claims cannot be made** from this dataset. Any future longitudinal work needs equipment held constant within-subject by design.
- **All findings condition on `session == 1` AND `qc_pass`.** Multi-session PD subjects are reserved for the longitudinal analysis and excluded from cross-sectional inference.

---

## Repository structure

```
parkinson-speech-analysis/
├── data/
│   ├── raw/                     (gitignored; place dataset here)
│   ├── interim/                 (subjects.csv, manifest.csv, 16 kHz audio)
│   └── processed/               (eGeMAPS + wav2vec2 parquet feature tables)
├── notebooks/
│   └── 04_eda.ipynb
├── src/
│   ├── preprocess.py            — audio resampling + QC + manifest
│   ├── features.py              — eGeMAPS extraction
│   ├── features_w2v2.py         — wav2vec2 embeddings
│   ├── eda.py                   
│   ├── stats.py                 
│   ├── models.py                — classifiers + leakage diagnostic
│   ├── longitudinal.py          — MixedLM + equipment-confound flag
│   └── interpret.py             — SHAP + bootstrap CIs + provenance
├── reports/
│   ├── figures/                 (eda/, longitudinal/, interpret/)
│   └── tables/                  (stats/, models/, longitudinal/, interpret/)
├── README.md
├── requirements.txt
└── LICENSE
```

---

## How to reproduce

### Environment

Python 3.12, conda. Tested on macOS 14 (M-series, MPS) and Linux. CUDA also supported for the wav2vec2 step; CPU works but is slow on long readings.

```bash
conda create -n parkinson-speech python=3.12 -y
conda activate parkinson-speech
pip install -r requirements.txt
```

### Data

Download the dataset from IEEE DataPort and place its three group folders under `data/raw/`:

```
data/raw/
├── 15 Young Healthy Control/
├── 22 Elderly Healthy Control/
└── 28 People with Parkinson's disease/
```

### Run

From the project root, in order:

```bash
python -m src.preprocess       
python -m src.features         
python -m src.features_w2v2     
python -m src.eda               
python -m src.stats             
python -m src.models            
python -m src.longitudinal      
python -m src.interpret         
```

Each script writes to `reports/` and is independent of every later step.

---

## Outputs

- **Cross-sectional stats:** `reports/tables/stats/{primary,exploratory}_{pd_vs_elderly,3group_b1b2pr1}.csv`, `sensitivity_pd_vs_elderly.csv`
- **Modeling:** `reports/tables/models/{per_task,pooled,leakage}_{cv_results,summary}.csv`
- **Longitudinal:** `reports/tables/longitudinal/{equipment_confound,pooled,per_task}.csv`, `reports/figures/longitudinal/trajectories.png`
- **Interpretability:** `reports/tables/interpret/{per_task_va1_shap_top,pooled_shap_top,leakage_shap_top,pooled_logreg_coefs_ci,feature_provenance}.csv`, `reports/figures/interpret/*.png`
- **EDA:** `reports/figures/eda/*.png` (cohort, PCA, UMAP, dysphonia markers)

---

## Methods notes

- **Feature extraction.** eGeMAPS via openSMILE (Eyben et al., 2016), 88 functionals per file. wav2vec2 via HuggingFace `facebook/wav2vec2-large-xlsr-53`, last-hidden-state mean+std pooled, 30s chunks with 5s overlap to fit MPS attention memory.
- **Stats.** OLS with `age + C(sex)` covariates; MixedLM via statsmodels; FDR via Benjamini-Hochberg per family.
- **Modeling.** All pipelines wrapped in `Pipeline([StandardScaler, clf])` — the scaler is fit per fold, no leakage. LogReg L2 (C=0.1, balanced class weights), RF (300 trees), GBM (200 trees, depth 3), XGBoost (300 trees, depth 4).
- **SHAP.** TreeExplainer per fold on test rows only — not whole-data SHAP. Same CV splits as `models.py` (`random_state=0`).
- **Bootstrap.** Cluster bootstrap by `subject_id` (B=1000), refit StandardScaler + L2 LogReg each iteration, 95% CI from 2.5/97.5 percentiles.

---

## License

Code: see `LICENSE`. Dataset is governed by the original IEEE DataPort terms — not redistributed here.
