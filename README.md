# Heart Failure Readmission & Resource Optimization

## Project Overview

This project develops a machine learning solution for healthcare resource optimization by focusing on heart failure readmission risk prediction and subsequent staffing resource allocation forecasting. By identifying patients at high risk of heart failure-related readmission and modeling specific staffing requirements, healthcare facilities can better allocate nursing resources, improve patient outcomes, and reduce costs associated with this high-burden condition.

## Business Context

Heart failure presents significant challenges for healthcare resource management:
- Affects approximately 6.2 million adults in the United States
- Responsible for over 1 million hospitalizations annually and costs hospitals more than $30 billion per year
- Has one of the highest 30-day readmission rates (~25%), resulting in significant Medicare penalties for hospitals
- Early intervention for high-risk patients can significantly reduce both readmission rates and associated staffing costs

## Data Source

This project uses the MIMIC-IV Clinical Database, a comprehensive dataset containing de-identified data of hospital stays for patients at Beth Israel Deaconess Medical Center, including:
- ICD codes for identifying heart failure patients
- Admission/discharge information to determine readmissions
- Vital signs, laboratory values, medications, and procedures
- Care unit information for staffing analysis
- Length of stay data critical for resource utilization calculation

## Project Structure

```
heart_failure_readmission/
│
├── data/                          # Data directory
│   ├── raw/                       # Raw MIMIC-IV data files
│   ├── interim/                   # Intermediate data that has been transformed
│   └── processed/                 # Final, canonical datasets for modeling
│
├── notebooks/                     # Jupyter notebooks for exploration and communication
│   ├── exploratory/               # Exploratory data analysis
│   └── reports/                   # Final analysis notebooks for reporting
│
├── src/                           # Source code for use in this project
│   ├── data/                      # Scripts to download or generate data
│   │   ├── make_dataset.py        # Script to identify heart failure patients
│   │   ├── extract_features.py    # Feature extraction module
│   │   └── preprocess.py          # Data preprocessing module
│   │
│   ├── models/                    # Scripts to train models and make predictions
│   │   ├── readmission_model.py   # Readmission prediction model
│   │   └── resource_model.py      # Resource utilization prediction model
│   │
│   ├── visualization/             # Scripts to create visualizations
│   └── utils/                     # Utility functions
│
├── tests/                         # Test files for code validation
├── config/                        # Configuration files
├── requirements.txt               # The requirements file for reproducing the environment
└── run_pipeline.sh                # Script to run the complete data pipeline
```

## Key Project Goals

1. Develop machine learning models to predict 30-day heart failure readmission risk
2. Create a staffing resource utilization forecasting system based on patient characteristics
3. Evaluate and compare different machine learning approaches for both problems
4. Build visualizations to communicate insights for clinical decision-making
5. Document a robust ML pipeline suitable for healthcare applications

## Technical Approach

### 1. Data Engineering & Preprocessing

- Identify heart failure patients using ICD-10 codes
- Determine index admissions and 30-day readmissions
- Extract demographic features, comorbidities, vital signs, lab values
- Process large MIMIC-IV tables with memory-efficient chunking techniques

### 2. Machine Learning Pipeline

- Readmission Prediction: Develop classification models (Logistic Regression, Random Forest, XGBoost)
- Staffing Resource Prediction: Create regression models to predict nursing hours by care level

### 3. Resource Optimization Framework

- Convert care levels to nursing ratios
- Calculate nursing hours based on length of stay and care level
- Estimate cost savings from prevented readmissions
- Calculate return on investment for intervention programs

## Setup and Installation

### Requirements

- Python 3.7+
- Pandas, NumPy, Scikit-learn
- Jupyter Notebook
- MIMIC-IV dataset access (requires credentialing)

### Installation

1. Clone this repository
```bash
git clone https://github.com/cyranothebard/heart_failure_readmission
cd heart_failure_readmission
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Set up MIMIC-IV data
   - Download MIMIC-IV dataset files and place in `data/raw/` directory
   - Alternatively, use Kaggle to download the dataset:
   ```bash
   python src/data/download_data.py data/raw
   ```

## Data Setup

This project follows a standard data science workflow with raw, interim, and processed data directories. However, actual data files are not committed to the repository due to their size.

### Initial Setup

1. Clone this repository
2. Set up your Python environment as described in the Setup section
3. Run the data directory setup script:
   ```bash
   python src/data/setup_data_structure.py

## Running the Pipeline

Execute the complete data processing pipeline:
```bash
bash run_pipeline.sh
```

Or run individual components:
```bash
# Identify heart failure patients and readmissions
python src/data/make_dataset.py data

# Extract features
python src/data/extract_features.py data

# Preprocess data
python src/data/preprocess.py data
```

## Model Development

After preprocessing, you can develop and evaluate models. The models and visualizations aren't stored in the repository but can be generated by running:
```bash
# Train and evaluate readmission prediction model
python src/models/readmission_model.py --data_dir data

# Train and evaluate resource utilization model
python src/models/resource_model.py --data_dir data
```

## Project Status

- [x] Data Engineering & Preprocessing
- [ ] Exploratory Data Analysis
- [ ] Readmission Prediction Model
- [ ] Resource Utilization Forecasting
- [ ] Visualization Dashboard
- [ ] Final Documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIMIC-IV database is provided by the MIT Laboratory for Computational Physiology
- This project is for educational purposes and is not intended for clinical use
```

