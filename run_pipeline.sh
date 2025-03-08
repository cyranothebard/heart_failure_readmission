# run_pipeline.sh

#!/bin/bash

# Set up environment
# echo "Setting up environment..."
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

# # Create directory structure if it doesn't exist
# echo "Creating directory structure..."
# mkdir -p data/raw data/interim data/processed
# mkdir -p notebooks/exploratory notebooks/reports

# # Download data
# echo "Step 1: Downloading MIMIC-IV data..."
# python src/data/download_data.py data/raw
# echo "Download complete. Checking files in data/raw:"
# ls -la data/raw

# Extract heart failure cohort
echo "Step 2: Identifying heart failure patients and readmissions..."
python src/data/make_dataset.py data
echo "After make_dataset.py, checking files in data/interim:"
ls -la data/interim

# Extract features
echo "Step 3: Extracting features..."
python src/data/extract_features.py data
echo "After extract_features.py, checking files in data/interim:"
ls -la data/interim

# Preprocess data
echo "Step 4: Preprocessing data..."
python src/data/preprocess.py data
echo "After preprocess.py, checking files in data/processed:"
ls -la data/processed

echo "Pipeline completed successfully!"