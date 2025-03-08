# src/data/make_dataset.py

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Add this at the top of make_dataset.py, extract_features.py, and preprocess.py
def ensure_correct_paths(data_dir):
    """Ensure that we're using the correct paths for raw, interim, and processed data"""
    import os
    
    # Normalize the base path
    base_dir = os.path.abspath(data_dir)

    # If the base_dir ends with 'raw', go up one level to the 'data' directory
    if os.path.basename(base_dir) == 'raw':
        base_dir = os.path.dirname(base_dir)  # Go up one level

    # Define subdirectories. If data_dir already ends with 'data', use it directly
    # Otherwise, assume we need to add subdirectories
    if os.path.basename(base_dir) == 'data':
        raw_dir = os.path.join(base_dir, 'raw')
        interim_dir = os.path.join(base_dir, 'interim')
        processed_dir = os.path.join(base_dir, 'processed')
    else:
        # If we were passed 'data/raw' directly
        raw_dir = base_dir
        interim_dir = os.path.join(os.path.dirname(base_dir), 'interim')
        processed_dir = os.path.join(os.path.dirname(base_dir), 'processed')

    # Ensure all directories exist
    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Make sure raw directory exists and has files
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    raw_files = os.listdir(raw_dir)
    if not raw_files:
        raise FileNotFoundError(f"No files found in raw data directory: {raw_dir}")
    
    print(f"Base directory: {base_dir}")
    print(f"Raw directory: {raw_dir}")
    print(f"Interim directory: {interim_dir}")
    print(f"Processed directory: {processed_dir}")
    print(f"Files in raw directory: {raw_files}")
    
    return base_dir, raw_dir, interim_dir, processed_dir

def load_mimic_tables(data_dir, tables=None):
    """
    Load specified MIMIC-IV tables from CSV files
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV CSV files
    tables : list or None
        List of tables to load; if None, loads all tables
        
    Returns:
    --------
    dict
        Dictionary of DataFrames, keyed by table name
    """
    if tables is None:
        tables = ['patients', 'admissions', 'diagnoses_icd', 
                  'labevents', 'prescriptions', 'transfers']
    
    data = {}
    for table in tables:
        file_path = os.path.join(data_dir, f'{table}.csv')
        try:
            data[table] = pd.read_csv(file_path)
            print(f"Loaded {table} with {len(data[table])} rows")
        except FileNotFoundError:
            print(f"Warning: Could not find {file_path}")
    
    return data

def identify_heart_failure_patients(diagnoses_df):
    """
    Identify patients with heart failure using ICD-10 codes (I50.*)
    
    Parameters:
    -----------
    diagnoses_df : DataFrame
        DataFrame containing diagnosis information
        
    Returns:
    --------
    DataFrame
        DataFrame with heart failure patients
    """
    # Check if input is a path to a CSV file
    if isinstance(diagnoses_df, str):
        # Process the file in chunks to save memory
        chunk_size = 100000  # Adjust based on your available memory
        hf_diagnoses_list = []
        
        # Read and process file in chunks
        for chunk in pd.read_csv(diagnoses_df, chunksize=chunk_size):
            # Filter for heart failure diagnoses (ICD-10 code starting with I50)
            hf_chunk = chunk[
                (chunk['icd_code'].astype(str).str.startswith('I50')) &
                (chunk['icd_version'] == 10)
            ]
            if not hf_chunk.empty:
                hf_diagnoses_list.append(hf_chunk)
        
        # Combine filtered chunks
        if hf_diagnoses_list:
            hf_diagnoses = pd.concat(hf_diagnoses_list, ignore_index=True)
        else:
            hf_diagnoses = pd.DataFrame()
    else:
        # Filter for heart failure diagnoses (ICD-10 code starting with I50)
        hf_diagnoses = diagnoses_df[
            (diagnoses_df['icd_code'].astype(str).str.startswith('I50')) &
            (diagnoses_df['icd_version'] == 10)
        ]
    
    print(f"Found {len(hf_diagnoses)} heart failure diagnoses")
    print(f"Found {hf_diagnoses['subject_id'].nunique()} unique patients with heart failure")
    
    return hf_diagnoses

def identify_readmissions(hf_patients, admissions_df, window_days=30):
    """
    Identify 30-day readmissions for heart failure patients
    
    Parameters:
    -----------
    hf_patients : DataFrame
        DataFrame containing heart failure patients
    admissions_df : DataFrame
        DataFrame containing admission information
    window_days : int
        Readmission window in days (default: 30)
        
    Returns:
    --------
    DataFrame
        DataFrame with readmission flags
    """
    # Get unique patient IDs with heart failure
    hf_patient_ids = hf_patients['subject_id'].unique()
    
    # Filter admissions to only include heart failure patients
    hf_admissions = admissions_df[admissions_df['subject_id'].isin(hf_patient_ids)].copy()
    
    # Convert admission and discharge times to datetime
    hf_admissions['admittime'] = pd.to_datetime(hf_admissions['admittime'])
    hf_admissions['dischtime'] = pd.to_datetime(hf_admissions['dischtime'])
    
    # Sort by patient ID and admission time
    hf_admissions.sort_values(['subject_id', 'admittime'], inplace=True)
    
    # Initialize readmission flag
    hf_admissions['is_readmission'] = False
    hf_admissions['days_until_readmission'] = np.nan
    
    # Group by patient ID to find readmissions
    for patient_id, group in hf_admissions.groupby('subject_id'):
        # Skip patients with only one admission
        if len(group) < 2:
            continue
        
        # Check each pair of consecutive admissions
        for i in range(len(group) - 1):
            discharge_time = group.iloc[i]['dischtime']
            next_admit_time = group.iloc[i+1]['admittime']
            
            # Calculate days between discharge and next admission
            days_between = (next_admit_time - discharge_time).total_seconds() / (24 * 3600)
            
            # If readmission occurred within the window
            if 0 < days_between <= window_days:
                # Mark the next admission as a readmission
                admission_id = group.iloc[i+1]['hadm_id']
                idx = hf_admissions[hf_admissions['hadm_id'] == admission_id].index
                hf_admissions.loc[idx, 'is_readmission'] = True
                hf_admissions.loc[idx, 'days_until_readmission'] = days_between
    
    readmission_count = hf_admissions['is_readmission'].sum()
    total_admissions = len(hf_admissions)
    print(f"Identified {readmission_count} readmissions out of {total_admissions} admissions ({readmission_count/total_admissions:.1%})")
    
    return hf_admissions

def main(data_dir):
    """
    Main data processing pipeline
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files or path to raw directory
    """
    # Get the correct paths
    base_dir, raw_dir, interim_dir, processed_dir = ensure_correct_paths(data_dir)
    
    # Instead of loading all tables into memory at once, process them one by one
    diagnoses_file = os.path.join(raw_dir, 'diagnoses_icd.csv')
    print(f"Processing diagnoses from: {diagnoses_file}")
    
    # Identify heart failure patients directly from file
    hf_diagnoses = identify_heart_failure_patients(diagnoses_file)
    
    # Load admissions data (this is usually smaller than diagnoses)
    admissions_file = os.path.join(raw_dir, 'admissions.csv')
    print(f"Loading admissions from: {admissions_file}")
    admissions_df = pd.read_csv(admissions_file)
    
    # Identify readmissions
    print("Identifying readmissions...")
    hf_admissions = identify_readmissions(hf_diagnoses, admissions_df)
    
    # Save intermediate result to the correct interim directory
    output_file = os.path.join(interim_dir, 'hf_admissions_with_readmissions.csv')
    print(f"Saving results to: {output_file}")
    hf_admissions.to_csv(output_file, index=False)
    
    print("Heart failure admissions data saved to interim folder")

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../../data'
    main(data_dir)