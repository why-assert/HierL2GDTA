"""
Standalone feature extraction script.
Deduplicates by protein, writes per-split subdirectories, supports resume.

Example:
python src/extract_features.py \
    --input data/processed/bindingdb.csv \
    --cache_dir feature_cache/bindingdb \
    --batch_size 64 \
    --device cuda

Notes:
1) CSV must contain smiles, protein, split columns.
2) Features are written to: cache_dir/<split>/a_{idx}.pt, b_{idx}.pt, c_{idx}.pt
3) Each unique protein sequence is encoded only once; other samples reuse it.
4) Resume support: progress is written after each complete sample.
"""

import os
import sys
from pathlib import Path

# Project root: this script lives in src/, so parent is project root
_project_root = Path(__file__).resolve().parent.parent
os.environ["TORCH_HOME"] = str(_project_root / "lib" / "esm")

import csv
import shutil
import argparse
from datetime import datetime
from typing import Optional, Dict, Tuple
import torch.nn.functional as F
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from encoders.esm_encoder import ESMEncoder
from encoders.chemberta_encoder import ChemBERTaEncoder

PROGRESS_FIELDS = [
    "timestamp",
    "idx",
    "split",
    "protein",
    "smiles",
    "a_path",
    "b_path",
    "c_path",
    "status",
]


def resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return project_root / p


def get_sequence_column(df: pd.DataFrame, preferred: str = "protein") -> str:
    """Return the protein sequence column, falling back to 'sequence'."""
    if preferred in df.columns:
        return preferred
    if "sequence" in df.columns:
        print("[Warning] 'protein' column not found; falling back to 'sequence'.")
        return "sequence"
    raise ValueError(
        "Input CSV must contain a 'protein' column (or a compatible 'sequence' column)."
    )


def split_aware_dir(base_cache_dir: Path, split_name: str) -> Path:
    """Append split_name unless base_cache_dir already ends with it."""
    split_name = str(split_name).strip().lower()
    if base_cache_dir.name.lower() == split_name:
        return base_cache_dir
    return base_cache_dir / split_name


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    """Try hard link first; fall back to copy."""
    if src == dst:
        return
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def progress_file_default(input_csv: Path) -> Path:
    """Default progress file next to the input CSV."""
    return input_csv.parent / f"{input_csv.stem}_extract_progress.csv"


def append_progress_row(progress_file: Path, record: Dict):
    file_exists = progress_file.exists()
    ensure_dir(progress_file.parent)
    with progress_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def load_progress_state(progress_file: Path) -> Tuple[set, Dict[str, Dict[str, Path]]]:
    """Return (completed_idx_set, protein_to_paths_map) from the progress file."""
    completed_idx = set()
    protein_cache: Dict[str, Dict[str, Path]] = {}

    if not progress_file.exists():
        return completed_idx, protein_cache

    try:
        df_prog = pd.read_csv(progress_file)
    except Exception as e:
        print(f"[Warning] Failed to read progress file, ignoring: {progress_file}\nReason: {e}")
        return completed_idx, protein_cache

    if "idx" not in df_prog.columns:
        return completed_idx, protein_cache

    for _, row in df_prog.iterrows():
        try:
            idx = int(row["idx"])
        except Exception:
            continue

        status = str(row.get("status", "")).strip().lower()
        if status.startswith("completed") or status.startswith("skipped"):
            completed_idx.add(idx)

        protein = str(row.get("protein", "")).strip()
        a_path = row.get("a_path", None)
        c_path = row.get("c_path", None)

        if protein and pd.notna(a_path) and pd.notna(c_path):
            a_p = Path(str(a_path))
            c_p = Path(str(c_path))
            if a_p.exists() and c_p.exists():
                if protein not in protein_cache:
                    protein_cache[protein] = {"a": a_p, "c": c_p}

    return completed_idx, protein_cache


def build_protein_context_tokens(feat, num_c_tokens=16):
    """
    Compress ESM residue-level features [protein_len, 1280]
    into ordered context tokens [context_len, 1280], context_len <= num_c_tokens.
    """
    if feat.ndim != 2:
        raise ValueError(
            f"Expected protein features [seq_len, embedding_dim], got {feat.shape}"
        )

    seq_len = feat.size(0)
    if seq_len <= 0:
        raise ValueError("Protein feature length must be > 0.")

    output_len = min(seq_len, num_c_tokens)
    context = F.adaptive_avg_pool1d(
        feat.transpose(0, 1).unsqueeze(0),
        output_size=output_len,
    )
    return context.squeeze(0).transpose(0, 1).contiguous()


def extract_and_cache_features_batched(
    data_df,
    esm_encoder,
    chem_encoder,
    cache_dir,
    progress_file: Path,
    batch_size: int = 64,
    protein_col: str = "protein",
    split_col: str = "split",
    num_c_tokens: int = 16,
):
    """
    Extract features in batches, reusing protein features by sequence.

    Outputs (split is required):
      cache_dir/<split>/a_{idx}.pt : residue-level protein features
      cache_dir/<split>/b_{idx}.pt : ligand features
      cache_dir/<split>/c_{idx}.pt : pooled protein context features
    """
    if split_col not in data_df.columns:
        raise ValueError(
            f"Data is missing the '{split_col}' column. "
            "Generate train/val/test splits first."
        )

    base_cache_dir = Path(cache_dir)
    ensure_dir(base_cache_dir)
    ensure_dir(progress_file.parent)

    print("Starting batched feature extraction (dedup by protein, split subdirs, resumable)...")

    completed_idx, protein_cache = load_progress_state(progress_file)
    if len(completed_idx) > 0:
        print(f"Restored {len(completed_idx)} completed records from progress file.")
    else:
        print("No resumable progress found; continuing from existing cache files.")

    total_rows = len(data_df)
    if total_rows == 0:
        print("Input data is empty; nothing to do.")
        return

    processed_new = 0
    skipped_existing = 0

    for start in tqdm(range(0, total_rows, batch_size), desc="Extracting", unit="batch"):
        end = min(start + batch_size, total_rows)
        batch_df = data_df.iloc[start:end].copy()

        need_rows = []

        for idx, row in batch_df.iterrows():
            split_name = str(row[split_col]).strip().lower()
            sample_dir = split_aware_dir(base_cache_dir, split_name)
            a_path = sample_dir / f"a_{idx}.pt"
            b_path = sample_dir / f"b_{idx}.pt"
            c_path = sample_dir / f"c_{idx}.pt"

            files_exist = a_path.exists() and b_path.exists() and c_path.exists()

            if files_exist:
                skipped_existing += 1
                if idx not in completed_idx:
                    record = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "idx": idx,
                        "split": split_name,
                        "protein": str(row[protein_col]).strip(),
                        "smiles": str(row["smiles"]).strip(),
                        "a_path": str(a_path.resolve()),
                        "b_path": str(b_path.resolve()),
                        "c_path": str(c_path.resolve()),
                        "status": "completed_existing",
                    }
                    append_progress_row(progress_file, record)
                    completed_idx.add(idx)

                protein_seq = str(row[protein_col]).strip()
                if protein_seq not in protein_cache:
                    protein_cache[protein_seq] = {"a": a_path, "c": c_path}
                continue

            need_rows.append((idx, row, sample_dir, a_path, b_path, c_path))

        if not need_rows:
            continue

        # Collect protein sequences not yet cached
        unseen_proteins = {}
        for idx, row, sample_dir, a_path, b_path, c_path in need_rows:
            protein_seq = str(row[protein_col]).strip()
            if protein_seq not in protein_cache and protein_seq not in unseen_proteins:
                unseen_proteins[protein_seq] = (idx, row, sample_dir)

        if unseen_proteins:
            prot_seqs = []
            prot_infos = []

            for protein_seq, (idx, row, sample_dir) in unseen_proteins.items():
                prot_seqs.append(protein_seq)
                prot_infos.append((protein_seq, idx, row, sample_dir))

            protein_feats_list = esm_encoder(prot_seqs)

            for (protein_seq, idx, row, sample_dir), feat in zip(prot_infos, protein_feats_list):
                ensure_dir(sample_dir)

                a_src = sample_dir / f"a_{idx}.pt"
                c_src = sample_dir / f"c_{idx}.pt"

                torch.save(feat, a_src)
                c = build_protein_context_tokens(feat, num_c_tokens=num_c_tokens)
                torch.save(c, c_src)

                protein_cache[protein_seq] = {"a": a_src, "c": c_src}

        # Ligand features are per-sample
        smiles_list = [str(row["smiles"]) for _, row, _, _, _, _ in need_rows]
        smiles_views_list = chem_encoder(smiles_list)
        smiles_feats_list = [views[0] for views in smiles_views_list]

        for local_i, (idx, row, sample_dir, a_path, b_path, c_path) in enumerate(need_rows):
            ensure_dir(sample_dir)

            protein_seq = str(row[protein_col]).strip()

            if protein_seq not in protein_cache:
                if a_path.exists() and c_path.exists():
                    protein_cache[protein_seq] = {"a": a_path, "c": c_path}
                else:
                    raise RuntimeError(
                        f"Protein not cached and no existing feature files found: "
                        f"{a_path} / {c_path}"
                    )

            src = protein_cache[protein_seq]

            if not a_path.exists():
                link_or_copy(src["a"], a_path)
            if not c_path.exists():
                link_or_copy(src["c"], c_path)
            if not b_path.exists():
                torch.save(smiles_feats_list[local_i], b_path)

            if a_path.exists() and b_path.exists() and c_path.exists():
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "idx": idx,
                    "split": str(row[split_col]).strip().lower(),
                    "protein": protein_seq,
                    "smiles": str(row["smiles"]).strip(),
                    "a_path": str(a_path.resolve()),
                    "b_path": str(b_path.resolve()),
                    "c_path": str(c_path.resolve()),
                    "status": "completed",
                }
                append_progress_row(progress_file, record)
                completed_idx.add(idx)
                processed_new += 1
            else:
                raise RuntimeError(f"Sample idx={idx} did not produce complete a/b/c files.")

        if (start // batch_size) % 10 == 0:
            print(
                f"Processed {min(end, total_rows)}/{total_rows} rows | "
                f"newly completed {processed_new} | "
                f"skipped existing {skipped_existing}"
            )

    print("Feature extraction finished.")
    print(f"Newly completed this run: {processed_new}")
    print(f"Skipped existing this run: {skipped_existing}")
    print(f"Progress file: {progress_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract protein and ligand features (dedup by protein, resumable)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV path (must contain smiles, protein, split columns).",
    )
    parser.add_argument(
        "--num_c_tokens",
        type=int,
        default=16,
        help="Max context tokens kept per protein, default 16.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="feature_cache/bindingdb",
        help="Feature cache root, e.g. feature_cache/bindingdb.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Extraction batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--esm_variant",
        type=str,
        default="normal",
        choices=["light", "normal"],
        help="ESM model variant.",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        default="split",
        help="Split column name; must exist in the CSV.",
    )
    parser.add_argument(
        "--protein_col",
        type=str,
        default="protein",
        help="Protein sequence column; falls back to 'sequence'.",
    )
    parser.add_argument(
        "--progress_file",
        type=str,
        default=None,
        help="Progress file path; defaults to a file next to the input CSV.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    data_file = resolve_path(args.input, project_root)
    cache_dir = resolve_path(args.cache_dir, project_root)

    if args.progress_file is None:
        progress_file = progress_file_default(data_file)
    else:
        progress_file = resolve_path(args.progress_file, project_root)

    df = pd.read_csv(data_file)

    required_cols = ["smiles"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input file must contain column: {required_cols}")

    # split column is required
    if args.split_col not in df.columns:
        raise SystemExit(
            f"[Error] Input CSV is missing the '{args.split_col}' column.\n"
            f"Generate train/val/test splits first.\n"
            f"CSV columns: {list(df.columns)}"
        )

    protein_col = get_sequence_column(df, preferred=args.protein_col)

    print(f"Rows: {len(df)}")
    print(f"Protein column: {protein_col}")
    print(f"Cache dir: {cache_dir}")
    print(f"Progress file: {progress_file}")

    print("Initializing encoders...")

    # Resolve ChemBERTa snapshot dir automatically (no hardcoded commit hash)
    _chemberta_snapshots = (
        project_root
        / "lib"
        / "ChemBERTa"
        / "models"
        / "models--DeepChem--ChemBERTa-77M-MTR"
        / "snapshots"
    )
    _snapshot_dirs = [d for d in _chemberta_snapshots.iterdir() if d.is_dir()]
    if not _snapshot_dirs:
        raise FileNotFoundError(
            f"No ChemBERTa snapshot found in {_chemberta_snapshots}. "
            "Please download the model first."
        )
    local_chemberta_path = _snapshot_dirs[0]

    chem_encoder = ChemBERTaEncoder(
        model_name=str(local_chemberta_path),
        device=device,
        batch_size=32,
        augment=0,
        cache=True,
        local_files_only=True,
    )

    esm_encoder = ESMEncoder(
        variant=args.esm_variant,
        device=device,
        batch_size=4,
        cache=True,
    )

    extract_and_cache_features_batched(
        df,
        esm_encoder,
        chem_encoder,
        cache_dir=cache_dir,
        progress_file=progress_file,
        batch_size=args.batch_size,
        protein_col=protein_col,
        split_col=args.split_col,
        num_c_tokens=args.num_c_tokens,
    )


if __name__ == "__main__":
    main()