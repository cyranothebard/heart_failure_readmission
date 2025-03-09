# src/data/preprocess.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Add this at the top of make_dataset.py, extract_features.py, and preprocess.py
def ensure_correct_paths(data_dir):
    """
    Ensure that we're using the correct paths for raw, interim, and processed data
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
        
    Returns:
    --------
    base_dir, raw_dir, interim_dir, processed_dir
    """
    import os
    
    # Normalize the base path
    base_dir = os.path.abspath(data_dir)
    
     # If data_dir already ends with 'data', use it directly
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

def create_final_dataset(data_dir):
    """
    Create the final dataset by merging patient and admission features

    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
        
    Returns:
    --------
    DataFrame
        The final dataset with all features
    """
    # Get the correct paths
    base_dir, raw_dir, interim_dir, processed_dir = ensure_correct_paths(data_dir)
    
    # Check for required files
    required_files = [
        os.path.join(interim_dir, 'hf_admissions_with_readmissions.csv'),
        os.path.join(interim_dir, 'hf_patient_features.csv'),
        os.path.join(interim_dir, 'hf_admission_features.csv')
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: The following required files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nContents of interim directory:")
        try:
            print(os.listdir(interim_dir))
        except Exception as e:
            print(f"Error listing interim directory: {e}")
        
        raise FileNotFoundError(f"Missing required files. Please run the previous pipeline steps.")
    
    # Load data
    print("Loading intermediate files...")
    hf_admissions = pd.read_csv(required_files[0])
    patient_features = pd.read_csv(required_files[1])
    admission_features = pd.read_csv(required_files[2])

    # Check if 'is_readmission' exists in both dataframes to prevent duplication
    if 'is_readmission' in hf_admissions.columns and 'is_readmission' in admission_features.columns:
        # Compare the 'is_readmission' columns
        comparison = (hf_admissions[['hadm_id', 'is_readmission']].set_index('hadm_id') ==
                      admission_features[['hadm_id', 'is_readmission']].set_index('hadm_id'))['is_readmission']
        
        if not comparison.all():
            raise ValueError("The 'is_readmission' column differs between hf_admissions and admission_features.  "
                             "Resolve this discrepancy before proceeding.")
        
        # Remove duplicate target column
        admission_features = admission_features.drop('is_readmission', axis=1)
    
    # Merge all features
    print("Merging feature sets...")
    merged_data = pd.merge(hf_admissions, patient_features, on='subject_id', how='left')
    merged_data = pd.merge(merged_data, admission_features, on='hadm_id', how='left')
    
    print(f"Final dataset has {len(merged_data)} rows and {len(merged_data.columns)} columns")
    
    return merged_data

def handle_missing_values(df, numerical_cols, strategy='median'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with missing values
    numerical_cols : list
        List of numerical column names
    strategy : str, default='median'
        Imputation strategy for numerical features
        
    Returns:
    --------
    DataFrame with imputed values
    """
    # Create a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # For numerical columns, use the specified strategy
    if numerical_cols:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # For non-numerical columns, fill with most frequent value
    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    if categorical_cols:
        df_imputed[categorical_cols] = df_imputed[categorical_cols].fillna(df_imputed[categorical_cols].mode().iloc[0])
    
    return df_imputed

def normalize_numerical_features(df, numerical_cols):
    """
    Normalize numerical features
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    numerical_cols : list
        List of numerical column names
        
    Returns:
    --------
    DataFrame with normalized features
    """
    # Create a copy to avoid modifying the original
    df_normalized = df.copy()
    
    # Apply StandardScaler to numerical columns
    if numerical_cols:
        scaler = StandardScaler()
        df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_normalized

def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    categorical_cols : list
        List of categorical column names
        
    Returns:
    --------
    DataFrame with encoded features, encoder object
    """
    # Create a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Apply one-hot encoding to categorical columns
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first') # changed parameter from sparse to sparce_output, as sparse was depreciated
        encoded_features = encoder.fit_transform(df[categorical_cols])
        
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(
            encoded_features, 
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
        
        # Drop original categorical columns
        df_encoded = df_encoded.drop(categorical_cols, axis=1)
        
        # Concatenate encoded features
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        return df_encoded, encoder
    
    return df_encoded, None

def prepare_data_for_modeling(df, target_col='is_readmission', test_size=0.2, random_state=42):
    """
    Prepare data for modeling by splitting into features and target,
    and then into training and testing sets
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    target_col : str, default='is_readmission'
        Target column name
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    # Separate features and target
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in DataFrame.")
        print("Available columns:")
        for col in df.columns:
            print(f"- {col}")
        raise ValueError(f"Target column '{target_col}' not found.")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"Positive class (readmissions) in training: {sum(y_train)}/{len(y_train)} ({sum(y_train)/len(y_train):.1%})")
    print(f"Positive class (readmissions) in testing: {sum(y_test)}/{len(y_test)} ({sum(y_test)/len(y_test):.1%})")
    
    return X_train, X_test, y_train, y_test

def main(data_dir):
    """
    Main preprocessing pipeline
    
    Parameters:
    -----------
    data_dir : str
        Directory containing MIMIC-IV files
    """
    # Get the correct paths
    base_dir, raw_dir, interim_dir, processed_dir = ensure_correct_paths(data_dir)
    
    # Create final dataset
    final_data = create_final_dataset(data_dir)
    
    # Define column types
    # These will need to be adjusted based on the actual data
    id_cols = ['subject_id', 'hadm_id']
    target_col = 'is_readmission'
    
    # Assuming these are in the final dataset - adjust as needed
    numerical_cols = ['anchor_age', 'total_los_days', 'icu_los_days', 'non_icu_los_days',
                     'sodium_mean', 'potassium_mean', 'bnp_mean', 'creatinine_mean', 
                     'hgb_mean', 'troponin_mean']
    
    categorical_cols = ['gender', 'insurance']
    
    # Exclude columns that won't be used in modeling
    exclude_cols = ['admittime', 'dischtime', 'deathtime', 'days_until_readmission']
    feature_cols = [col for col in final_data.columns if col not in id_cols + [target_col] + exclude_cols]
    
    # Adjust numerical and categorical columns to only include feature columns
    numerical_cols = [col for col in numerical_cols if col in feature_cols]
    categorical_cols = [col for col in categorical_cols if col in feature_cols]
    
    # Handle missing values
    data_imputed = handle_missing_values(final_data, numerical_cols)
    
    # Normalize numerical features
    data_normalized = normalize_numerical_features(data_imputed, numerical_cols)
    
    # Encode categorical features
    data_processed, encoder = encode_categorical_features(data_normalized, categorical_cols)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(data_processed, target_col)
    
    # Save processed data
    processed_dir = os.path.join(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save full processed dataset
    data_processed.to_csv(os.path.join(processed_dir, 'hf_data_processed.csv'), index=False)
    
    # Save train/test splits
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    print("Preprocessing complete. Files saved to processed folder.")

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../../data'
    main(data_dir)