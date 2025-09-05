# src/data/extract_features.py

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime, timedelta

def ensure_correct_paths(data_dir):
    """Ensure that we're using the correct paths for raw, interim, and processed data"""
    import os
    
    # Normalize the base path
    base_dir = os.path.abspath(data_dir)
    
    # If the base_dir ends with 'raw', go up one level to the 'data' directory
    if os.path.basename(base_dir) == 'raw':
        base_dir = os.path.dirname(base_dir)  # Go up one level
    
    # Define subdirectories
    raw_dir = os.path.join(base_dir, 'raw')
    interim_dir = os.path.join(base_dir, 'interim')
    processed_dir = os.path.join(base_dir, 'processed')
    
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
    print(f"Files in raw directory: {len(raw_files)}")
    
    return base_dir, raw_dir, interim_dir, processed_dir

def extract_demographics_chunked(admissions_file, patients_file, output_file, chunk_size=10000):
    """
    Extract demographic features in a memory-efficient way
    
    Parameters:
    -----------
    admissions_file : str
        Path to heart failure admissions file
    patients_file : str
        Path to patients file
    output_file : str
        Path to output file for demographic features
    chunk_size : int
        Size of chunks to process at once
    """
    print(f"Extracting demographics from {admissions_file} and {patients_file}")
    
    # First, extract unique subject IDs from admissions
    print("Extracting unique subject IDs from admissions file...")
    unique_subjects = set()
    
    for chunk in pd.read_csv(admissions_file, chunksize=chunk_size, usecols=['subject_id']):
        unique_subjects.update(chunk['subject_id'].unique())
    
    print(f"Found {len(unique_subjects)} unique subjects")
    
    # Create a DataFrame with just the subject IDs
    subjects_df = pd.DataFrame({'subject_id': list(unique_subjects)})
    
    # Process patients file in chunks
    print("Processing patients file in chunks...")
    patients_chunks = []
    
    for chunk in pd.read_csv(patients_file, chunksize=chunk_size):
        # Filter to relevant patients
        filtered_chunk = chunk[chunk['subject_id'].isin(unique_subjects)].copy()
        
        if not filtered_chunk.empty:
            # Keep only necessary columns to save memory
            if 'gender' in filtered_chunk.columns:
                keep_cols = ['subject_id', 'gender']
                if 'anchor_age' in filtered_chunk.columns:
                    keep_cols.append('anchor_age')
                
                filtered_chunk = filtered_chunk[keep_cols]
                patients_chunks.append(filtered_chunk)
                
                # Report progress
                if len(patients_chunks) % 10 == 0:
                    print(f"Processed {len(patients_chunks)} chunks...")
    
    # Combine all chunks
    if patients_chunks:
        demographics = pd.concat(patients_chunks, ignore_index=True)
        
        # Remove duplicates if any
        demographics = demographics.drop_duplicates(subset=['subject_id'])
        
        print(f"Final demographics shape: {demographics.shape}")
        
        # Save to file
        demographics.to_csv(output_file, index=False)
        print(f"Demographics saved to {output_file}")
        
        return demographics
    else:
        print("No matching patients found!")
        # Create an empty file with just the headers
        pd.DataFrame(columns=['subject_id', 'gender']).to_csv(output_file, index=False)
        return pd.DataFrame()

def extract_comorbidities_chunked(admissions_file, diagnoses_file, output_file, chunk_size=10000):
    """
    Extract comorbidity features in a memory-efficient way
    
    Parameters:
    -----------
    admissions_file : str
        Path to heart failure admissions file
    diagnoses_file : str
        Path to diagnoses file
    output_file : str
        Path to output file for comorbidity features
    chunk_size : int
        Size of chunks to process at once
    """
    print(f"Extracting comorbidities from {admissions_file} and {diagnoses_file}")
    
    # First, extract unique subject IDs from admissions
    print("Extracting unique subject IDs from admissions file...")
    unique_subjects = set()
    
    for chunk in pd.read_csv(admissions_file, chunksize=chunk_size, usecols=['subject_id']):
        unique_subjects.update(chunk['subject_id'].unique())
    
    print(f"Found {len(unique_subjects)} unique subjects")
    
    # Define comorbidity ICD codes
    # Include both ICD-9 and ICD-10 codes for broader coverage
    comorbidity_map = {
        'hypertension': {
            'icd9': ['401', '402', '403', '404', '405'],
            'icd10': ['I10', 'I11', 'I12', 'I13', 'I15']
        },
        'diabetes': {
            'icd9': ['250'],
            'icd10': ['E10', 'E11', 'E13']
        },
        'kidney_disease': {
            'icd9': ['585', '586'],
            'icd10': ['N18', 'N19']
        },
        'copd': {
            'icd9': ['496'],
            'icd10': ['J44']
        },
        'atrial_fibrillation': {
            'icd9': ['4273'],
            'icd10': ['I48']
        },
        'coronary_artery_disease': {
            'icd9': ['414'],
            'icd10': ['I25']
        },
        'obesity': {
            'icd9': ['278'],
            'icd10': ['E66']
        },
        'anemia': {
            'icd9': ['280', '281', '282', '283', '284', '285'],
            'icd10': ['D50', 'D51', 'D52', 'D53']
        }
    }
    
    # Initialize dictionary to track comorbidities
    patient_comorbidities = {subject_id: {comorbidity: 0 for comorbidity in comorbidity_map} for subject_id in unique_subjects}
    
    # Process diagnoses file in chunks
    print("Processing diagnoses file in chunks...")
    chunks_processed = 0
    matches_found = 0
    
    for chunk in pd.read_csv(diagnoses_file, chunksize=chunk_size):
        # Filter to relevant patients
        filtered_chunk = chunk[chunk['subject_id'].isin(unique_subjects)].copy()
        
        chunks_processed += 1
        if chunks_processed % 10 == 0:
            print(f"Processed {chunks_processed} chunks...")
        
        if filtered_chunk.empty:
            continue
        
        # Check each comorbidity
        for comorbidity, codes in comorbidity_map.items():
            # Check ICD-9 codes
            if 'icd_version' in filtered_chunk.columns and 'icd_code' in filtered_chunk.columns:
                for code_type, code_list in codes.items():
                    version = 9 if code_type == 'icd9' else 10
                    
                    for code in code_list:
                        # Find diagnoses that match this code
                        matches = filtered_chunk[
                            (filtered_chunk['icd_version'] == version) &
                            (filtered_chunk['icd_code'].astype(str).str.startswith(code))
                        ]
                        
                        if not matches.empty:
                            matches_found += len(matches)
                            
                            # Mark patients with this comorbidity
                            for subject_id in matches['subject_id'].unique():
                                if subject_id in patient_comorbidities:
                                    patient_comorbidities[subject_id][comorbidity] = 1
            else:
                print(f"WARNING: Required columns not found in diagnoses file: {filtered_chunk.columns.tolist()}")
    
    print(f"Processed {chunks_processed} chunks, found {matches_found} matching diagnoses")
    
    # Convert dictionary to DataFrame
    comorbidities_data = []
    for subject_id, comorbidities in patient_comorbidities.items():
        row = {'subject_id': subject_id}
        row.update(comorbidities)
        comorbidities_data.append(row)
    
    comorbidities_df = pd.DataFrame(comorbidities_data)
    
    print(f"Final comorbidities shape: {comorbidities_df.shape}")
    
    # Save to file
    comorbidities_df.to_csv(output_file, index=False)
    print(f"Comorbidities saved to {output_file}")
    
    return comorbidities_df

def create_minimal_admission_features(admissions_file, output_file):
    """
    Create a minimal set of admission features 
    
    Parameters:
    -----------
    admissions_file : str
        Path to heart failure admissions file
    output_file : str
        Path to output file for admission features
    """
    print(f"Creating minimal admission features from {admissions_file}")
    
    # Read admissions file
    hf_admissions = pd.read_csv(admissions_file)
    
    # Get unique admission IDs
    hadm_ids = hf_admissions['hadm_id'].unique()
    
    # Create a minimal DataFrame with just the admission IDs
    admission_features = pd.DataFrame({'hadm_id': hadm_ids})
    
    # Add some basic features if available
    if 'is_readmission' in hf_admissions.columns:
        # Group by hadm_id and aggregate
        readmission_status = hf_admissions.groupby('hadm_id')['is_readmission'].max().reset_index()
        admission_features = pd.merge(admission_features, readmission_status, on='hadm_id', how='left')
    
    print(f"Created minimal admission features with shape: {admission_features.shape}")
    
    # Save to file
    admission_features.to_csv(output_file, index=False)
    print(f"Minimal admission features saved to {output_file}")
    
    return admission_features

def main(data_dir):
    """
    Main feature extraction pipeline with memory optimization
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
    """
    # Get the correct paths
    base_dir, raw_dir, interim_dir, processed_dir = ensure_correct_paths(data_dir)
    
    # Set file paths
    admissions_file = os.path.join(interim_dir, 'hf_admissions_with_readmissions.csv')
    patients_file = os.path.join(raw_dir, 'patients.csv')
    diagnoses_file = os.path.join(raw_dir, 'diagnoses_icd.csv')
    
    # Set output paths
    demographics_file = os.path.join(interim_dir, 'hf_patient_features.csv')
    comorbidities_file = os.path.join(interim_dir, 'hf_comorbidities.csv')
    admission_features_file = os.path.join(interim_dir, 'hf_admission_features.csv')
    
    # Check if admissions file exists
    if not os.path.exists(admissions_file):
        print(f"ERROR: Admissions file not found: {admissions_file}")
        print("Make sure to run make_dataset.py first")
        return
    
    # Extract demographics
    print("\n=== Extracting demographics ===")
    extract_demographics_chunked(admissions_file, patients_file, demographics_file)
    
    # Force garbage collection
    gc.collect()
    
    # Extract comorbidities
    print("\n=== Extracting comorbidities ===")
    try:
        extract_comorbidities_chunked(admissions_file, diagnoses_file, comorbidities_file)
    except Exception as e:
        print(f"Error in comorbidity extraction: {e}")
        # Continue with other steps
    
    # Force garbage collection
    gc.collect()
    
    # Create minimal admission features to satisfy pipeline dependency
    print("\n=== Creating minimal admission features ===")
    create_minimal_admission_features(admissions_file, admission_features_file)
    
    # Try to merge patient features
    print("\n=== Merging patient features ===")
    try:
        # Check if both files exist
        if os.path.exists(demographics_file) and os.path.exists(comorbidities_file):
            # Read files
            demographics = pd.read_csv(demographics_file)
            comorbidities = pd.read_csv(comorbidities_file)
            
            # Merge
            patient_features = pd.merge(demographics, comorbidities, on='subject_id', how='outer')
            
            # Save merged file
            patient_features.to_csv(demographics_file, index=False)
            print(f"Merged patient features saved to {demographics_file}")
        else:
            print("One or more patient feature files missing, skipping merge")
    except Exception as e:
        print(f"Error merging patient features: {e}")
    
    print("\nFeature extraction complete!")

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../../data'
    main(data_dir)