"""
Davis dataset preprocessing script.

Converts raw Davis CSV to a filtered CSV with pKd values.

Usage:
    python davis/davis.py \
        --input davis.csv \
        --output davis_filtered.csv \
        --max-smiles-length 300

Automatically detects column names (drug_id, protein_id, smiles, protein,
affinity, etc.), drops missing values, canonicalizes SMILES, deduplicates
by drug_id + protein_id, and converts affinity to pKd.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
tqdm.pandas()

def print_filter_result(step_name, before_count, after_count):
    """
    Print the number of removed and retained records.
    """
    removed_count = before_count - after_count

    if before_count == 0:
        percentage = 0.0
    else:
        percentage = removed_count / before_count * 100

    print(
        f"[{step_name}] "
        f"removed: {removed_count}, "
        f"retained: {after_count}/{before_count}, "
        f"removed percentage: {percentage:.3f}%"
    )

def find_column(df, candidates, column_type):
    """
    Find the first existing column from candidates.
    """
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Cannot find {column_type} column.\n"
        f"Supported names: {candidates}\n"
        f"Current columns: {list(df.columns)}"
    )

def load_csv(input_file):
    """
    Load the input CSV file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file does not exist: {input_file}"
        )

    df = pd.read_csv(input_file)

    print(f"Input file: {input_file}")
    print(f"Original records: {len(df)}")
    print(f"Original columns: {list(df.columns)}")

    if len(df) == 0:
        raise ValueError("The input CSV file is empty.")

    return df

def select_and_rename_columns(df):
    """
    Select and rename the required columns.

    Expected columns from the previously generated CSV:

        drug_id
        drug
        protein_id
        protein
        affinity
    """

    # For the generated Davis CSV, "drug" is the SMILES column.
    smiles_column = find_column(
        df,
        [
            "drug",
            "smiles",
            "SMILES",
            "drug_smiles",
            "Drug",
        ],
        "SMILES",
    )

    protein_column = find_column(
        df,
        [
            "protein",
            "Target",
            "target",
            "Protein",
            "protein_sequence",
        ],
        "protein",
    )

    affinity_column = find_column(
        df,
        [
            "affinity",
            "Kd",
            "kd",
            "Ki",
            "ki",
            "Y",
            "y",
        ],
        "affinity",
    )

    drug_id_column = find_column(
        df,
        [
            "drug_id",
            "Drug_ID",
            "compound_id",
            "compound",
            "drug_index",
        ],
        "drug ID",
    )

    protein_id_column = find_column(
        df,
        [
            "protein_id",
            "Protein_ID",
            "target_id",
            "Target_ID",
            "protein_index",
        ],
        "protein ID",
    )

    result = df[
        [
            drug_id_column,
            protein_id_column,
            smiles_column,
            protein_column,
            affinity_column,
        ]
    ].copy()

    result = result.rename(
        columns={
            drug_id_column: "drug_id",
            protein_id_column: "protein_id",
            smiles_column: "smiles",
            protein_column: "protein",
            affinity_column: "affinity",
        }
    )

    print(f"Drug ID column: {drug_id_column}")
    print(f"Protein ID column: {protein_id_column}")
    print(f"SMILES column: {smiles_column}")
    print(f"Protein sequence column: {protein_column}")
    print(f"Affinity column: {affinity_column}")

    return result

def filter_missing_values(df):
    """
    Remove records with missing IDs, SMILES, protein or affinity.
    """
    before_count = len(df)

    missing_mask = (
        df["drug_id"].isna()
        | df["protein_id"].isna()
        | df["smiles"].isna()
        | df["protein"].isna()
        | df["affinity"].isna()
    )

    df = df.loc[~missing_mask].copy()
    df.reset_index(drop=True, inplace=True)

    print_filter_result(
        "Missing value check",
        before_count,
        len(df),
    )

    return df

def clean_ids_and_sequences(df):
    """
    Strip whitespace from IDs and protein sequences.
    """
    before_count = len(df)

    for column in ["drug_id", "protein_id", "protein"]:
        df[column] = df[column].astype(str).str.strip()

    invalid_mask = (
        (df["drug_id"] == "")
        | (df["protein_id"] == "")
        | (df["protein"] == "")
    )

    df = df.loc[~invalid_mask].copy()
    df.reset_index(drop=True, inplace=True)

    print_filter_result(
        "ID and protein text check",
        before_count,
        len(df),
    )

    return df

def standardize_smiles(df, max_smiles_length=300):
    """
    Standardize SMILES with RDKit.

    Invalid SMILES and SMILES longer than max_smiles_length
    are removed.
    """
    before_count = len(df)

    def convert_one_smiles(smiles):
        try:
            smiles = str(smiles).strip()

            if not smiles:
                return np.nan

            molecule = Chem.MolFromSmiles(smiles)

            if molecule is None:
                return np.nan

            canonical_smiles = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )

            if not canonical_smiles:
                return np.nan

            if len(canonical_smiles) > max_smiles_length:
                return np.nan

            return canonical_smiles

        except Exception:
            return np.nan

    df["smiles"] = df["smiles"].progress_apply(
        convert_one_smiles
    )

    invalid_mask = df["smiles"].isna()

    df = df.loc[~invalid_mask].copy()
    df.reset_index(drop=True, inplace=True)

    print_filter_result(
        "SMILES standardization",
        before_count,
        len(df),
    )

    return df

def convert_affinity_to_numeric(df):
    """
    Convert affinity values to numeric values.
    """
    before_count = len(df)

    df["affinity"] = pd.to_numeric(
        df["affinity"],
        errors="coerce",
    )

    invalid_mask = df["affinity"].isna()

    df = df.loc[~invalid_mask].copy()
    df.reset_index(drop=True, inplace=True)

    print_filter_result(
        "Affinity numeric check",
        before_count,
        len(df),
    )

    return df

def filter_non_positive_affinity(df):
    """
    Remove affinity values less than or equal to zero.
    """
    before_count = len(df)

    invalid_mask = df["affinity"] <= 0

    df = df.loc[~invalid_mask].copy()
    df.reset_index(drop=True, inplace=True)

    print_filter_result(
        "Affinity positive-value check",
        before_count,
        len(df),
    )

    return df

def remove_duplicate_records_by_ids(df):
    """
    Remove duplicate records only when both IDs are identical.

    Duplicate key:
        drug_id + protein_id

    The first record is retained.
    If the same pair has different affinity values,
    the first record is also retained.
    """
    before_count = len(df)

    df = df.drop_duplicates(
        subset=["drug_id", "protein_id"],
        keep="first",
    ).copy()

    df.reset_index(drop=True, inplace=True)

    print_filter_result(
        "Duplicate ID pair check",
        before_count,
        len(df),
    )

    return df

def convert_to_pkd(df):
    """
    Convert Kd in nM to pKd.

    pKd = 9 - log10(Kd[nM])
    """
    df["pKd"] = 9.0 - np.log10(df["affinity"])

    return df

def process_dataset(
    input_file,
    output_file,
    max_smiles_length=300,
):
    """
    Process the complete dataset.
    """
    df = load_csv(input_file)

    df = select_and_rename_columns(df)

    df = filter_missing_values(df)

    df = clean_ids_and_sequences(df)

    df = standardize_smiles(
        df,
        max_smiles_length=max_smiles_length,
    )

    df = convert_affinity_to_numeric(df)

    df = filter_non_positive_affinity(df)

    # Only drug_id and protein_id are used to identify duplicates.
    df = remove_duplicate_records_by_ids(df)

    df = convert_to_pkd(df)

    output_df = df[
        [
            "drug_id",
            "protein_id",
            "smiles",
            "protein",
            "pKd",
        ]
    ].copy()

    output_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Processing completed.")
    print(f"Output file: {output_file}")
    print(f"Final records: {len(output_df)}")
    print(f"Output columns: {list(output_df.columns)}")
    print()
    print(output_df.head())

def main():
    parser = argparse.ArgumentParser(
        description="Process Davis CSV and convert Kd to pKd."
    )

    parser.add_argument(
        "--input",
        default="davis.csv",
        help="Input CSV file. Default: davis.csv",
    )

    parser.add_argument(
        "--output",
        default="davis_pkd.csv",
        help="Output CSV file. Default: davis_pkd.csv",
    )

    parser.add_argument(
        "--max-smiles-length",
        type=int,
        default=300,
        help="Maximum SMILES length. Default: 300",
    )

    args = parser.parse_args()

    process_dataset(
        input_file=args.input,
        output_file=args.output,
        max_smiles_length=args.max_smiles_length,
    )

if __name__ == "__main__":
    main()