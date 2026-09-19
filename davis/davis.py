"""
Davis dataset preprocessing script (PaddleHelix raw format).

Converts the raw Davis dataset from PaddleHelix (ligands_can.txt +
proteins.txt + affinity matrix) into a filtered CSV with pKd values.

Usage:
    python davis/davis.py \
        --input_dir davis/davis \
        --output davis/davis_filtered.csv

Input directory should contain:
    ligands_can.txt   - one canonical SMILES per line
    proteins.txt      - one protein sequence per line
    Y                 - affinity matrix (Kd in nM), shape (n_drugs, n_proteins)
                        (can be .txt or no extension)

Output columns: drug_id, protein_id, smiles, protein, pKd
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
tqdm.pandas()


def print_filter_result(step_name, before_count, after_count):
    """Print the number of removed and retained records."""
    removed = before_count - after_count
    pct = removed / before_count * 100 if before_count > 0 else 0.0
    print(
        f"[{step_name}] "
        f"removed: {removed}, retained: {after_count}/{before_count}, "
        f"removed percentage: {pct:.3f}%"
    )


def load_davis_raw(input_dir):
    """
    Load raw Davis dataset from PaddleHelix format.

    Reads ligands_can.txt, proteins.txt, and the affinity matrix Y,
    then converts to long-format DataFrame.
    """
    input_dir = Path(input_dir)

    # --- Load ligands ---
    ligands_file = input_dir / "ligands_can.txt"
    if not ligands_file.exists():
        for alt in ["ligands.txt"]:
            if (input_dir / alt).exists():
                ligands_file = input_dir / alt
                break

    if not ligands_file.exists():
        raise FileNotFoundError(
            f"Cannot find ligands file in {input_dir}. "
            f"Expected: ligands_can.txt or ligands.txt"
        )

    # Try JSON dict format first (PaddleHelix format: {drug_id: smiles})
    try:
        import json
        with open(ligands_file, "r") as f:
            ligands_dict = json.load(f)
        if isinstance(ligands_dict, dict):
            drug_ids = list(ligands_dict.keys())
            ligands = list(ligands_dict.values())
            print(f"Loaded {len(ligands)} ligands from {ligands_file.name} (JSON dict format)")
        else:
            raise ValueError("Not a dict")
    except Exception:
        # Fallback: one SMILES per line
        with open(ligands_file, "r") as f:
            ligands = [line.strip() for line in f if line.strip()]
        drug_ids = [f"D{i+1}" for i in range(len(ligands))]
        print(f"Loaded {len(ligands)} ligands from {ligands_file.name} (line format)")

    # --- Load proteins ---
    proteins_file = input_dir / "proteins.txt"
    if not proteins_file.exists():
        raise FileNotFoundError(
            f"Cannot find proteins.txt in {input_dir}"
        )

    # Try JSON dict format first (PaddleHelix format: {protein_id: sequence})
    try:
        import json
        with open(proteins_file, "r") as f:
            proteins_dict = json.load(f)
        if isinstance(proteins_dict, dict):
            protein_ids = list(proteins_dict.keys())
            proteins = list(proteins_dict.values())
            print(f"Loaded {len(proteins)} proteins from proteins.txt (JSON dict format)")
        else:
            raise ValueError("Not a dict")
    except Exception:
        # Fallback: one sequence per line
        with open(proteins_file, "r") as f:
            proteins = [line.strip() for line in f if line.strip()]
        protein_ids = [f"P{i+1}" for i in range(len(proteins))]
        print(f"Loaded {len(proteins)} proteins from proteins.txt (line format)")

    # --- Load affinity matrix ---
    y_file = None
    for candidate in ["Y", "Y.txt", "mat_drug_protein.txt", "affinity.txt"]:
        if (input_dir / candidate).exists():
            y_file = input_dir / candidate
            break

    if y_file is None:
        raise FileNotFoundError(
            f"Cannot find affinity matrix in {input_dir}. "
            f"Expected: Y, Y.txt, mat_drug_protein.txt, or affinity.txt"
        )

    # Try multiple formats: pickle (Python2 numpy), npy, text
    try:
        # Try pickle first (PaddleHelix Davis uses Python2 pickle format)
        with open(y_file, "rb") as f:
            import pickle
            affinity_matrix = pickle.load(f, encoding="latin1")
        if not isinstance(affinity_matrix, np.ndarray):
            affinity_matrix = np.array(affinity_matrix)
    except Exception:
        try:
            # Try np.load (npy format)
            affinity_matrix = np.load(str(y_file), allow_pickle=True)
        except Exception:
            try:
                # Try text format
                affinity_matrix = np.loadtxt(str(y_file))
            except Exception:
                # Try pandas as last resort
                affinity_matrix = pd.read_csv(
                    y_file, sep=None, engine="python", header=None
                ).values

    print(f"Loaded affinity matrix from {y_file.name}, shape: {affinity_matrix.shape}")

    # Verify dimensions
    n_drugs, n_prots = affinity_matrix.shape
    if n_drugs != len(ligands):
        print(
            f"Warning: matrix rows ({n_drugs}) != ligands count ({len(ligands)})"
        )
    if n_prots != len(proteins):
        print(
            f"Warning: matrix cols ({n_prots}) != proteins count ({len(proteins)})"
        )

    # --- Convert to long format ---
    print("Converting matrix to long format...")
    rows = []
    for i in range(n_drugs):
        for j in range(n_prots):
            affinity = affinity_matrix[i, j]
            rows.append({
                "drug_id": drug_ids[i],
                "protein_id": protein_ids[j],
                "smiles": ligands[i] if i < len(ligands) else "",
                "protein": proteins[j] if j < len(proteins) else "",
                "affinity": affinity,
            })

    df = pd.DataFrame(rows)
    print(f"Total pairs: {len(df)}")

    return df


def standardize_smiles(df, max_smiles_length=300):
    """Standardize SMILES with RDKit. Remove invalid / too long ones."""
    before_count = len(df)

    def convert_one(smiles):
        try:
            smiles = str(smiles).strip()
            if not smiles:
                return np.nan
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.nan
            canonical = Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            )
            if not canonical or len(canonical) > max_smiles_length:
                return np.nan
            return canonical
        except Exception:
            return np.nan

    df["smiles"] = df["smiles"].progress_apply(convert_one)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    print_filter_result("SMILES standardization", before_count, len(df))
    return df


def filter_missing_and_invalid(df):
    """Remove rows with missing protein sequence or invalid affinity."""
    before_count = len(df)

    # Check protein
    df["protein"] = df["protein"].astype(str).str.strip()
    df = df[df["protein"] != ""]

    # Check affinity is numeric and positive
    df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")
    df = df.dropna(subset=["affinity"])
    df = df[df["affinity"] > 0]

    df = df.reset_index(drop=True)
    print_filter_result(
        "Missing / invalid value check", before_count, len(df)
    )
    return df


def convert_to_pkd(df):
    """Convert Kd (nM) to pKd: pKd = 9 - log10(Kd_nM)"""
    df["pKd"] = 9.0 - np.log10(df["affinity"])
    return df


def process_dataset(input_dir, output_file, max_smiles_length=300):
    """Full Davis preprocessing pipeline."""
    print("=" * 60)
    print("Davis Dataset Preprocessing")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print()

    # 1. Load raw data
    df = load_davis_raw(input_dir)

    # 2. Filter missing / invalid values
    df = filter_missing_and_invalid(df)

    # 3. Standardize SMILES
    df = standardize_smiles(df, max_smiles_length=max_smiles_length)

    # 4. Convert Kd to pKd
    df = convert_to_pkd(df)

    # 5. Output
    output_df = df[["drug_id", "protein_id", "smiles", "protein", "pKd"]].copy()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    output_df.to_csv(output_file, index=False)

    print()
    print("=" * 60)
    print("Processing completed.")
    print(f"Output file: {output_file}")
    print(f"Final records: {len(output_df)}")
    print(f"Output columns: {list(output_df.columns)}")
    print(f"pKd range: {output_df['pKd'].min():.3f} ~ {output_df['pKd'].max():.3f}")
    print("=" * 60)
    print()
    print(output_df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Process Davis dataset (PaddleHelix raw format) to pKd CSV."
    )
    parser.add_argument(
        "--input_dir",
        default="davis/davis",
        help="Input directory containing ligands_can.txt, proteins.txt, and Y. "
             "Default: davis/davis",
    )
    parser.add_argument(
        "--output",
        default="davis/davis_filtered.csv",
        help="Output CSV file. Default: davis/davis_filtered.csv",
    )
    parser.add_argument(
        "--max-smiles-length",
        type=int,
        default=300,
        help="Maximum SMILES length. Default: 300",
    )
    args = parser.parse_args()

    process_dataset(
        input_dir=args.input_dir,
        output_file=args.output,
        max_smiles_length=args.max_smiles_length,
    )


if __name__ == "__main__":
    main()
