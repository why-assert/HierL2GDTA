"""
BindingDB raw data preprocessing script.

Converts BindingDB_All.tsv to a filtered CSV with pKd values.

Usage:
    python bindingdb/bindingdb.py <input.tsv> [output.csv]

Filters applied:
- Single-chain proteins only
- Kd range: 0.01 nM ~ 1e7 nM
- Mammals/birds kept; viruses kept only if temp in 33~40°C
- Protein length: 50 ~ 2560
- SMILES canonicalized, max length 300
- Duplicates (same SMILES + UniProt) aggregated via 20th percentile Kd

Output columns: smiles, protein, uniprot_id, Kd, pKd
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

CFG = {
    # Kd in nM
    'Kd_range': [1e-2, 1e7],
    # Experimental temperature in Celsius
    'T_range': [33, 40],
    # Protein sequence length
    'protein_len_range': [50, 2560],
    # Approximate scale factor for <Kd and >Kd
    'approx_scale': 3.16,
}

def report_filter(name, before_count, after_count):
    removed = before_count - after_count
    print(f'{name}: removed {removed:,}, remaining {after_count:,}')

def filter_rows(name, df, keep_mask):
    before_count = len(df)
    result = df.loc[keep_mask].copy()
    result.reset_index(drop=True, inplace=True)
    report_filter(name, before_count, len(result))
    return result

def is_missing(value):
    if pd.isna(value):
        return True
    return str(value).strip() == ''

def parse_numeric_value(value):
    """Parse numeric values, <value, >value, and values with units."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if match is None:
        return np.nan

    number = float(match.group(0))
    if text.startswith('<'):
        number /= CFG['approx_scale']
    elif text.startswith('>'):
        number *= CFG['approx_scale']
    return number

def convert_numeric_column(df, column):
    df = df.copy()
    df[column] = df[column].apply(parse_numeric_value)
    return df

def read_bindingdb(filename):
    """Read BindingDB TSV and extract required columns."""
    print(f'Reading BindingDB: {filename}')

    header = pd.read_csv(filename, sep='\t', nrows=0, low_memory=False).columns.tolist()
    print(f'Detected original columns: {len(header):,}')

    def find_column(candidates):
        for column in candidates:
            if column in header:
                return column
        return None

    smiles_col = find_column([
        'Ligand SMILES',
        'Ligand SMILES (Canonical)',
        'Ligand SMILES (Isomeric)',
    ])
    kd_col = find_column(['Kd (nM)', 'Kd'])
    temperature_col = find_column(['Temp (C)', 'Temperature (C)', 'Temperature'])
    chain_count_col = find_column([
        'Number of Protein Chains in Target (>1 implies a multichain complex)',
        'Number of Protein Chains in Target',
    ])
    sequence_col = find_column([
        'BindingDB Target Chain Sequence 1',
        'Target Chain Sequence 1',
        'BindingDB Target Chain Sequence',
        'Target Chain Sequence',
    ])
    organism_col = find_column([
        'Target Source Organism According to Curator or DataSource',
        'Target Source Organism',
        'Organism',
    ])
    swissprot_id_col = find_column([
        'UniProt (SwissProt) Primary ID of Target Chain 1',
        'UniProt (SwissProt) Primary ID of Target Chain',
        'UniProt (SwissProt) Primary ID',
    ])
    trembl_id_col = find_column([
        'UniProt (TrEMBL) Primary ID of Target Chain 1',
        'UniProt (TrEMBL) Primary ID of Target Chain',
        'UniProt (TrEMBL) Primary ID',
    ])

    required_columns = {
        'Ligand SMILES': smiles_col,
        'Kd (nM)': kd_col,
        'BindingDB Target Chain Sequence 1': sequence_col,
        'Target Source Organism': organism_col,
        'Number of Protein Chains in Target': chain_count_col,
    }
    missing_required = [name for name, value in required_columns.items() if value is None]
    if missing_required:
        raise ValueError(
            'Missing required fields: ' + ', '.join(missing_required) +
            '\nRun the following command to check the full header:\n'
            "head -n 1 BindingDB_All.tsv | tr '\\t' '\\n' | less"
        )

    selected_columns = [smiles_col, kd_col, sequence_col, organism_col, chain_count_col]
    for column in [temperature_col, swissprot_id_col, trembl_id_col]:
        if column is not None and column not in selected_columns:
            selected_columns.append(column)

    print('Columns read:')
    for column in selected_columns:
        print(f'  - {column}')

    df = pd.read_csv(filename, sep='\t', usecols=selected_columns, low_memory=False)

    rename_map = {
        smiles_col: 'smiles',
        kd_col: 'Kd',
        sequence_col: 'protein',
        organism_col: 'organism',
        chain_count_col: 'n_chains',
    }
    if temperature_col is not None:
        rename_map[temperature_col] = 'T'
    if swissprot_id_col is not None:
        rename_map[swissprot_id_col] = 'swissprot_id'
    if trembl_id_col is not None:
        rename_map[trembl_id_col] = 'trembl_id'

    df.rename(columns=rename_map, inplace=True)

    if 'T' not in df.columns:
        df['T'] = np.nan
        print('Warning: Temperature column not found, creating empty T column.')
    if 'swissprot_id' not in df.columns:
        df['swissprot_id'] = pd.NA
    if 'trembl_id' not in df.columns:
        df['trembl_id'] = pd.NA

    for column in ['swissprot_id', 'trembl_id']:
        df[column] = df[column].astype('string').str.strip()
        df[column] = df[column].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'NA': pd.NA})

    df['uniprot_id'] = df['swissprot_id'].fillna(df['trembl_id'])
    df.drop(columns=['swissprot_id', 'trembl_id'], inplace=True)

    print(f'Successfully read records: {len(df):,}')
    print(f'Records with Chain 1 protein sequence: {df["protein"].notna().sum():,}')
    print(f'Records with Chain 1 UniProt ID: {df["uniprot_id"].notna().sum():,}')

    return df

def filter_multichain(df):
    """Keep only single-chain targets."""
    chain_count = pd.to_numeric(df['n_chains'], errors='coerce')
    keep_mask = chain_count.eq(1)
    return filter_rows('Multichain complex filtering', df, keep_mask)

def filter_missing_protein_ligand(df):
    protein_missing = df['protein'].isna() | df['protein'].astype('string').str.strip().eq('')
    ligand_missing = df['smiles'].isna() | df['smiles'].astype('string').str.strip().eq('')
    keep_mask = ~(protein_missing | ligand_missing)
    return filter_rows('Missing protein/ligand filtering', df, keep_mask)

def filter_kd(df):
    min_kd, max_kd = CFG['Kd_range']
    keep_mask = (
        df['Kd'].notna()
        & np.isfinite(df['Kd'])
        & (df['Kd'] >= min_kd)
        & (df['Kd'] <= max_kd)
    )
    return filter_rows('Invalid or missing Kd filtering', df, keep_mask)

def classify_organism(value):
    """
    Classify organism based on Target Source Organism text.
    Returns 'animal', 'virus', 'other', or 'unknown'.
    """
    if pd.isna(value):
        return 'unknown'
    text = str(value).strip().lower()
    if not text or text in {'nan', 'none', 'na', 'n/a', '-'}:
        return 'unknown'

    # Mammals
    mammal_patterns = [
        r'\bhomo sapiens\b', r'\bhuman\b',
        r'\bmus musculus\b', r'\bmouse\b', r'\bmice\b', r'\brattus norvegicus\b', r'\brat\b',
        r'\bmacaca\b', r'\bmacaca mulatta\b', r'\bmacaca fascicularis\b', r'\bchlorocebus\b',
        r'\bpapio\b', r'\bpan troglodytes\b', r'\bpan paniscus\b', r'\bpongo\b',
        r'\bcallithrix\b', r'\bmonkey\b', r'\bchimpanzee\b', r'\bprimate\b',
        r'\boryctolagus cuniculus\b', r'\brabbit\b',
        r'\bcanis lupus familiaris\b', r'\bdog\b',
        r'\bfelis catus\b', r'\bcat\b',
        r'\bsus scrofa\b', r'\bpig\b',
        r'\bbos taurus\b', r'\bcattle\b', r'\bcow\b', r'\bbovine\b',
        r'\bovis aries\b', r'\bsheep\b',
        r'\bcapra hircus\b', r'\bgoat\b',
        r'\bequus caballus\b', r'\bhorse\b',
        r'\bmammal\b', r'\bmammalia\b', r'哺乳',
    ]
    if any(re.search(pattern, text) for pattern in mammal_patterns):
        return 'animal'

    # Birds
    bird_patterns = [
        r'\bgallus gallus\b', r'\bchicken\b',
        r'\bmeleagris gallopavo\b', r'\bturkey\b',
        r'\banas platyrhynchos\b', r'\bduck\b', r'\bgoose\b',
        r'\bcoturnix\b', r'\bquail\b',
        r'\btaeniopygia guttata\b', r'\bzebra finch\b',
        r'\bavian\b', r'\bbird\b', r'\baves\b', r'鸟',
    ]
    if any(re.search(pattern, text) for pattern in bird_patterns):
        return 'animal'

    # Viruses
    virus_patterns = [
        r'\bpararnavirae\b', r'\borthornavirae\b', r'\bribozyviria\b', r'\bshotokuvirae\b',
        r'\bvaridnaviria\b', r'\bherpesvirales\b',
        r'\bcoronavirus\b', r'\bsars[- ]?cov\b', r'\bsars[- ]?cov[- ]?2\b',
        r'\bsevere acute respiratory syndrome coronavirus 2\b',
        r'\bmiddle east respiratory syndrome\b', r'\bmers[- ]?cov\b',
        r'\bhuman immunodeficiency virus\b', r'\bhiv[- ]?[12]?\b', r'\bretrovirus\b',
        r'\binfluenza\b', r'\binfluenzavirus\b', r'\borthomyxoviridae\b',
        r'\bhepatitis [abcde] virus\b', r'\bhepatitis [abcde]\b', r'\bhbv\b', r'\bhcv\b',
        r'\bherpesvirus\b', r'\bherpes simplex\b', r'\bcytomegalovirus\b', r'\bepstein[- ]barr\b',
        r'\bzika\b', r'\bdengue\b', r'\bflavivirus\b', r'\bebola\b', r'\bfilovirus\b',
        r'\bnipah\b', r'\bmeasles\b', r'\brabies\b', r'\bparamyxovirus\b',
        r'\bpoxvirus\b', r'\bpapillomavirus\b', r'\badenovirus\b', r'\benterovirus\b',
        r'\bpoliovirus\b', r'\bnorovirus\b', r'\bvirus\b', r'\bviridae\b', r'病毒',
    ]
    if any(re.search(pattern, text) for pattern in virus_patterns):
        return 'virus'

    return 'other'

def filter_organism_and_temperature(df):
    """
    Organism and temperature filtering:
    1. Mammals/birds: keep.
    2. Viruses: keep only if temperature in CFG['T_range'].
    3. Other/unknown: remove.
    """
    if df.empty:
        print('Organism/temperature filtering: input empty, skipping.')
        return df.copy()

    min_temp, max_temp = CFG['T_range']
    df = df.copy()
    df['organism_class'] = df['organism'].apply(classify_organism)

    animal_count = df['organism_class'].eq('animal').sum()
    virus_count = df['organism_class'].eq('virus').sum()
    other_count = df['organism_class'].eq('other').sum()
    unknown_count = df['organism_class'].eq('unknown').sum()

    print(
        f'Organism classification: mammals/birds {animal_count:,}, '
        f'viruses {virus_count:,}, other {other_count:,}, unknown {unknown_count:,}'
    )

    temperature_ok = (
        df['T'].notna()
        & np.isfinite(df['T'])
        & (df['T'] >= min_temp)
        & (df['T'] <= max_temp)
    )
    keep_mask = df['organism_class'].eq('animal') | (df['organism_class'].eq('virus') & temperature_ok)

    result = filter_rows('Organism/temperature filtering', df, keep_mask)
    result.drop(columns=['organism_class'], inplace=True)
    return result

def filter_protein_length(df):
    min_len, max_len = CFG['protein_len_range']
    sequence = df['protein'].astype('string').str.replace(r'\s+', '', regex=True)
    sequence_length = sequence.str.len()
    keep_mask = sequence.notna() & sequence_length.between(min_len, max_len, inclusive='both')
    df = df.copy()
    df['protein'] = sequence
    return filter_rows('Protein sequence length filtering', df, keep_mask)

def canonicalize_smiles(smiles):
    if pd.isna(smiles):
        return np.nan
    text = str(smiles).strip()
    if not text:
        return np.nan
    try:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return np.nan
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if canonical is None or len(canonical) > 300:
            return np.nan
        return canonical
    except Exception:
        return np.nan

def filter_and_canonicalize_smiles(df):
    """Canonicalize SMILES. Use plain pandas apply for compatibility."""
    if df.empty:
        print('SMILES parsing/too long filtering: input empty, skipping.')
        return df.copy()
    df = df.copy()
    df['smiles'] = df['smiles'].apply(canonicalize_smiles)
    keep_mask = df['smiles'].notna()
    return filter_rows('SMILES parsing/too long filtering', df, keep_mask)

def percentile_value(data, percentile):
    """Compute percentile (equivalent to weighted_percentile with unit weights)."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.nan
    data = np.sort(data)
    n = len(data)
    positions = (np.arange(1, n + 1) - 0.5) / n
    return float(np.interp(percentile / 100.0, positions, data))

def aggregate_kd_constants(df):
    """Aggregate duplicate Kd measurements for the same SMILES and UniProt ID."""
    if df.empty:
        return pd.DataFrame(columns=['smiles', 'protein', 'uniprot_id', 'Kd'])

    grouped = df.groupby(['smiles', 'uniprot_id'], sort=False).groups
    records = []

    for (smiles, uniprot_id), indices in tqdm(grouped.items(), desc='Aggregating duplicate drug-target records'):
        group = df.iloc[indices]
        kd_values = group['Kd'].to_numpy(dtype=float)
        kd = percentile_value(kd_values, 20)
        records.append({
            'smiles': smiles,
            'protein': group.iloc[0]['protein'],
            'uniprot_id': uniprot_id,
            'Kd': kd,
        })

    return pd.DataFrame(records)

def convert_kd_to_pkd(kd):
    # Kd in nM: pKd = 9 - log10(Kd[nM])
    return 9.0 - np.log10(kd)

def run(source_file, output_file='bindingdb_filtered.csv'):
    df = read_bindingdb(source_file)
    print(f'Original records: {len(df):,}')

    df = filter_multichain(df)
    df = filter_missing_protein_ligand(df)

    df = convert_numeric_column(df, 'Kd')
    df = convert_numeric_column(df, 'T')
    df = filter_kd(df)

    print('\nMost common Target Source Organism after Kd filtering:')
    if df.empty:
        print('No records.')
    else:
        print(
            df['organism']
            .fillna('MISSING')
            .astype(str)
            .value_counts()
            .head(30)
            .to_string()
        )
    print()

    df = filter_organism_and_temperature(df)
    df = filter_protein_length(df)
    df = filter_and_canonicalize_smiles(df)

    result = aggregate_kd_constants(df)

    if len(result) == 0:
        print('No records to output after filtering.')
        result = pd.DataFrame(columns=['smiles', 'protein', 'uniprot_id', 'Kd', 'pKd'])
    else:
        result['pKd'] = result['Kd'].apply(convert_kd_to_pkd)
        result = result[['smiles', 'protein', 'uniprot_id', 'Kd', 'pKd']]

    result.to_csv(output_file, index=False)

    print(f'Final records: {len(result):,}')
    print(f'Output file: {output_file}')

if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print(
            'Usage:\n'
            '  python bindingdb.py <input.tsv> [output.csv]\n\n'
            'Example:\n'
            '  python bindingdb.py BindingDB_All.tsv bindingdb_filtered.csv'
        )
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else 'bindingdb_filtered.csv'
    run(input_file, output_file)