# HierL2GDTA

Hierarchical L2G Drug-Target Affinity (DTA) prediction model.

Based on the ESM-2 protein sequence encoder and ChemBERTa ligand SMILES encoder, it predicts drug-target affinity through a hierarchical local-to-global (L2G) graph structure.

---

## Directory Structure

```
HierL2GDTA/
├── bindingdb/                  # BindingDB data preprocessing
│   └── bindingdb.py
├── davis/                      # Davis data preprocessing
│   └── davis.py
├── lib/                        # Third-party pre-trained model weights (offline loading)
│   ├── ChemBERTa/models/       # ChemBERTa-77M-MTR (ligand encoder)
│   └── esm/hub/checkpoints/    # ESM-2 650M (protein encoder)
├── src/
│   ├── encoders/
│   │   ├── esm_encoder.py      # ESM-2 protein sequence encoder
│   │   └── chemberta_encoder.py # ChemBERTa ligand SMILES encoder
│   ├── model.py                # HierL2G model definition
│   ├── config.py               # Default model configuration
│   ├── data.py                 # Dataset and DataLoader
│   ├── engine.py               # Training / evaluation loops
│   ├── train.py                # Main training entry point
│   ├── extract_features.py     # Pre-extract features (supports resumable extraction)
│   ├── checkpoint.py           # Checkpoint save / load
│   ├── metrics.py              # Evaluation metrics
│   └── utils.py                # Utility functions
├── tools/
│   └── split_dataset.py        # Random dataset split (7:2:1)
└── README.md
```

---

## Step 1: Install Dependencies

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (recommended; both training and feature extraction support GPU)

### Install

```bash
pip install torch transformers fair-esm pandas numpy rdkit tqdm scipy
```

> The ESM Python package (`fair-esm`) must be installed via pip. Model weights need to be downloaded separately (see next step).

---

## Step 2: Download Pre-trained Models

The project uses two pre-trained models, both loaded in offline mode (`local_files_only=True`). Download them to the corresponding paths under `lib/`.

### 2.1 ESM-2 650M (Protein Encoder)

| Item | Value |
|------|-------|
| Model | `esm2_t33_650M_UR50D` |
| Parameters | 650M / 33 layers |
| Output dim | 1280 per residue |
| Path | `lib/esm/hub/checkpoints/esm2_t33_650M_UR50D.pt` |

**Download:**

```bash
mkdir -p lib/esm/hub/checkpoints

wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt \
    -P lib/esm/hub/checkpoints/
```

Alternatively, download manually from the [fair-esm GitHub Release page](https://github.com/facebookresearch/esm#available-models).

### 2.2 ChemBERTa-77M-MTR (Ligand Encoder)

| Item | Value |
|------|-------|
| Model | `DeepChem/ChemBERTa-77M-MTR` |
| Parameters | 77M |
| Output dim | 384 per token |
| Path | `lib/ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/` |

**Method 1: Python script (recommended)**

```python
from huggingface_hub import snapshot_download
from pathlib import Path

model_dir = Path("lib/ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/snapshots/main")
model_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="DeepChem/ChemBERTa-77M-MTR",
    local_dir=str(model_dir),
    local_dir_use_symlinks=False
)
```

**Method 2: git clone (requires git-lfs)**

```bash
git lfs install
git clone https://huggingface.co/DeepChem/ChemBERTa-77M-MTR \
    lib/ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/snapshots/main
```

> The code auto-scans subdirectories under `snapshots/` and does not depend on a specific commit hash directory name. Any subdirectory name works.

### 2.3 Verify Models

After downloading, the directory structure should look like:

```
lib/
├── esm/hub/checkpoints/
│   └── esm2_t33_650M_UR50D.pt
└── ChemBERTa/models/models--DeepChem--ChemBERTa-77M-MTR/
    └── snapshots/
        └── main/         # or any commit hash directory
            ├── config.json
            ├── pytorch_model.bin  # or model.safetensors
            ├── tokenizer.json
            ├── vocab.json
            ├── merges.txt
            └── ...
```

---

## Step 3: Download Raw Datasets

### 3.1 Davis Dataset

```bash
# Download to davis/ directory
wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/dti_datasets/davis_v1.tgz -P davis/

# Extract
tar -xzf davis/davis_v1.tgz -C davis/
```

This extracts to `davis/davis/` containing three files:

| File | Format | Description |
|------|--------|-------------|
| `ligands_can.txt` | JSON dict | `{drug_id: canonical_SMILES}` pairs, 68 drugs |
| `proteins.txt` | JSON dict | `{protein_id: sequence}` pairs, 442 proteins |
| `Y` | Python2 pickle | Affinity matrix (Kd in nM), shape `[68, 442]` |

The preprocessing script auto-detects these formats.

### 3.2 BindingDB Dataset (202604 version)

Download `BindingDB_All.tsv` from the [BindingDB download page](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) and place it in the `bindingdb/` directory:

```bash
# Download the 202604 TSV release (or check the download page for current version)
curl -L -o bindingdb/BindingDB_All_tsv.zip \
    "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202604_tsv.zip"

# Extract
unzip bindingdb/BindingDB_All_tsv.zip -d bindingdb/
```

This produces `bindingdb/BindingDB_All.tsv`.

> **Note**: BindingDB releases are versioned by date (e.g., `202604` = April 2026). Older versions may be removed over time. Check the [official download page](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) if the link above is no longer valid.

---

## Step 4: Standard Training Pipeline

### 4.1 BindingDB

```bash
# 1. Preprocess
python bindingdb/bindingdb.py bindingdb/BindingDB_All.tsv bindingdb/bindingdb_filtered.csv

# 2. Split (7:2:1, output to data/)
python tools/split_dataset.py --input bindingdb/bindingdb_filtered.csv --output_dir data/bindingdb

# 3. Extract features
python src/extract_features.py \
    --input data/bindingdb/splited.csv \
    --cache_dir feature_cache/bindingdb \
    --device cuda

# 4. Train
python src/train.py \
    --data_file data/bindingdb/splited.csv \
    --cache_root feature_cache/bindingdb \
    --save_dir checkpoints/bindingdb \
    --use_amp
```

### 4.2 Davis

```bash
# 1. Preprocess (PaddleHelix raw format → filtered CSV with pKd)
python davis/davis.py --input_dir davis/davis --output davis/davis_filtered.csv

# 2. Split (7:2:1, output to data/)
python tools/split_dataset.py --input davis/davis_filtered.csv --output_dir data/davis

# 3. Extract features
python src/extract_features.py \
    --input data/davis/splited.csv \
    --cache_dir feature_cache/davis \
    --device cuda

# 4. Train
python src/train.py \
    --data_file data/davis/splited.csv \
    --cache_root feature_cache/davis \
    --save_dir checkpoints/davis \
    --use_amp
```

### 4.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 16 | Batch size |
| `--epochs` | 50 | Maximum training epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--max_seq_len` | 1400 | Maximum sequence length |
| `--max_c_len` | 16 | Protein context token count |
| `--target_col` | auto-detect | Target column (pKd / pKi / affinity) |
| `--use_amp` | off | Mixed precision training |

**Optimizer:** AdamW + ReduceLROnPlateau (factor=0.5, patience=3) + MSE Loss

**Outputs (in `save_dir`):** `best_model.pt` / `last_model.pt` / `test_metrics.json` / `test_predictions.csv`

> Feature extraction supports resumable extraction and protein deduplication. Training supports resuming from `last_model.pt` automatically.

---

## Step 5: Custom Dataset Training

If you have your own DTA dataset (with ligand SMILES, protein sequences, and affinity labels), you can train directly without using the bindingdb/davis preprocessing scripts.

### 5.1 Data Format

Prepare a CSV file with **at least the following columns**:

| Column | Required | Description |
|--------|----------|-------------|
| `smiles` | ✅ | Ligand SMILES string |
| `protein` | ✅ | Protein amino acid sequence |
| `pKd` (or `pKi` / `affinity`) | ✅ (training mode) | Affinity label, numeric |
| `split` | ❌ | Split marker (train / val / test); if absent, use `split_dataset.py` |

> Target column names are auto-detected **case-insensitively**: `pKd` first, then `pKi`, then `affinity`. You can also manually specify any column name with `--target_col`.

### 5.2 Quick Start (3 steps)

```bash
# Suppose your data is mydata/data.csv with smiles, protein, and pKd columns

# Step 1: Split dataset (7:2:1)
python tools/split_dataset.py \
    --input mydata/data.csv \
    --output_dir data/mydata

# Step 2: Extract features
python src/extract_features.py \
    --input data/mydata/splited.csv \
    --cache_dir feature_cache/mydata \
    --batch_size 64 \
    --device cuda

# Step 3: Train
python src/train.py \
    --data_file data/mydata/splited.csv \
    --cache_root feature_cache/mydata \
    --save_dir checkpoints/mydata \
    --use_amp \
    --epochs 50 \
    --batch_size 16
```

### 5.3 Advanced: Custom Splits

If your dataset already has predefined splits (e.g., protein-clustered split, drug-clustered split), add a `split` column to your CSV (values: `train` / `val` / `test`) and skip `split_dataset.py`, starting directly from feature extraction:

```bash
# CSV already has split column, extract features directly
python src/extract_features.py \
    --input mydata/data_with_split.csv \
    --cache_dir feature_cache/mydata \
    --device cuda

# Train directly
python src/train.py \
    --data_file mydata/data_with_split.csv \
    --cache_root feature_cache/mydata \
    --save_dir checkpoints/mydata \
    --use_amp
```

### 5.4 Advanced: Pre-extracted Features

If features are already extracted (containing `a_{idx}.pt`, `b_{idx}.pt`, `c_{idx}.pt`), you can train directly:

```bash
python src/train.py \
    --data_file mydata/splited.csv \
    --cache_root /path/to/feature_cache \
    --save_dir checkpoints/mydata \
    --use_amp
```

### 5.5 Data Recommendations

- **SMILES**: recommended length ≤ 300; very long sequences may be truncated
- **Protein sequences**: recommended length 50 ~ 1400 (adjustable via `--max_seq_len`)
- **Labels**: pKd / pKi format works best; typical range is 4~12
- **Dataset size**: at least 1000 training samples recommended to avoid overfitting

---

## Step 6: Prediction (Inference)

After training, use `src/predict.py` to predict affinity for new SMILES / protein pairs.

### 6.1 Input Format

CSV must contain at least `smiles` and `protein` columns. If a ground-truth label column exists (`pKd` / `pKi` / `affinity`), evaluation metrics are computed automatically.

**No `split` column required.**

### 6.2 Prediction Command

**Minimal usage:**

```bash
python src/predict.py \
    --input data/predict/my_data.csv \
    --model_path checkpoints/bindingdb/best_model.pt \
    --device cuda
```

**Full example:**

```bash
python src/predict.py \
    --input data/predict/candidates.csv \
    --model_path checkpoints/bindingdb/best_model.pt \
    --output predict/candidates_result.csv \
    --cache_dir feature_cache/predict/candidates \
    --target_col pKd \
    --device cuda \
    --batch_size 32 \
    --max_seq_len 1400 \
    --max_c_len 16
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--input` | ✅ | - | Input CSV path |
| `--model_path` | ✅ | - | Model checkpoint path (best_model.pt or last_model.pt) |
| `--output` | ❌ | `predict/<csv_name>_predictions.csv` | Output CSV path |
| `--cache_dir` | ❌ | `feature_cache/predict/<csv_filename>/` | Feature cache directory |
| `--target_col` | ❌ | auto-detect | Target column name (metrics computed if present) |
| `--device` | ❌ | `cuda` | Device: cuda / cpu |
| `--batch_size` | ❌ | 32 | Batch size |
| `--max_seq_len` | ❌ | 1400 | Maximum sequence length |
| `--max_c_len` | ❌ | 16 | Protein context token count |
| `--no_extract` | ❌ | off | Skip feature extraction (use when features are already cached) |

### 6.3 Output

The output CSV contains all columns from the input CSV plus a `prediction` column (predicted affinity value, denormalized to original scale).

If the input CSV has a label column, MSE / RMSE / MAE / R² / Pearson metrics are printed to the terminal.

**Recommended file layout for custom data:** place input data in `data/predict/` (e.g., `data/predict/candidates.csv`), features are cached to `feature_cache/predict/candidates/`, and predictions are output to `predict/candidates_predictions.csv`.

### 6.4 Features

- **Auto feature extraction**: automatically extracts and caches features on first run; subsequent predictions reuse the cache
- **Resumable extraction**: resumes from where it left off if interrupted
- **Protein deduplication**: same protein sequence is encoded only once
- **Auto denormalization**: predictions are automatically scaled back to original label space (mean/std read from checkpoint)
- **Optional metrics**: evaluation metrics computed automatically when ground-truth labels are available

---

## Model Architecture

The HierL2G model uses a hierarchical local-to-global (L2G) feature interaction strategy:

1. **Local encoding**: ESM-2 and ChemBERTa extract initial representations for proteins and ligands, respectively
2. **Local interaction**: multi-scale convolution kernels (1/3/5) + dilated convolution extract local structural patterns
3. **Hierarchical graph interaction**: multi-layer Transformer blocks implement hierarchical protein-ligand interaction
4. **Global aggregation**: attention pooling + fully connected layers output the affinity prediction

Default configuration:
- Hidden dimension: 640
- Transformer heads: 4
- Transformer blocks: 2
- GRU layers: 2
- Dropout: 0.15

---

## FAQ

**Q: Feature extraction is slow?**
A: Protein feature extraction is the bottleneck. Each unique protein sequence is encoded only once, so datasets with higher protein redundancy will be faster. You can also increase `--batch_size`.

**Q: How to resume interrupted training?**
A: Just re-run `train.py`; it automatically resumes from `last_model.pt` by default. Use `--no_resume` to disable.

**Q: How to resume interrupted feature extraction?**
A: Just re-run `extract_features.py`; it automatically skips already-completed samples.

**Q: Does it support pKi labels?**
A: Yes. Target column names are auto-detected case-insensitively (pKd / pKi / affinity). You can also manually specify with `--target_col`.

**Q: Does it work on Windows?**
A: The code is cross-platform, but hard links may fall back to file copying on some Windows filesystems (no impact on functionality, just uses more disk space). Linux is recommended.
