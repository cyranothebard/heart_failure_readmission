# src/data/download_data.py

import os
import kagglehub
import shutil

def download_mimic_data(output_dir):
    """
    Download the MIMIC-IV dataset from Kaggle
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the dataset files
    
    Returns:
    --------
    str
        Path to the downloaded files
    """
    print("Downloading MIMIC-IV dataset from Kaggle...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download latest version
    path = kagglehub.dataset_download("dipthith/data-dc")
    
    print(f"Files downloaded to: {path}")
    
    # Copy files to the raw data directory
    file_count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                source_file = os.path.join(root, file)
                dest_file = os.path.join(output_dir, file)
                shutil.copy2(source_file, dest_file)
                file_count += 1
                print(f"Copied: {file}")
    
    print(f"Total files copied: {file_count}")
    
    return output_dir

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '../../data/raw'
    download_mimic_data(output_dir)