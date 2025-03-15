"""
This script runs the complete modeling pipeline on the test subset of data.
It helps to identify and fix issues before running on the full dataset.
"""

import os
import argparse
import logging
import time
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_test_pipeline(data_dir='data', create_subset=True, subset_size=0.2, tune_models=False, output_dir=None, models='all'):
    """
    Run the model pipeline on test data
    
    Parameters:
    -----------
    data_dir : str
        Base directory for data
    create_subset : bool
        Whether to create a new test subset
    subset_size : float
        Proportion of data to include in subset
    tune_models : bool
        Whether to tune model hyperparameters
    output_dir : str, optional
        Directory to save test results
    models : str
        Which models to test ('all', 'readmission', or 'resource')
    """
    start_time = time.time()
    logging.info("Starting test pipeline...")
    
    # Ensure test directory exists
    test_dir = os.path.join(data_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        logging.info(f"Created test directory: {test_dir}")
    
    # Create test subset
    if create_subset:
        logging.info(f"Creating test subset (size={subset_size})...")
        from src.data.create_test_subset import create_subset
        create_subset(data_dir, subset_size=subset_size)
    
    # Define test data directory
    test_dir = os.path.join(data_dir, 'test')
    
    # Import model modules
    from src.models import readmission_model, resource_model
    
    readmission_results = None
    resource_results = None
    resource_summary = None
    
    # Run readmission model if specified
    if models in ['all', 'readmission']:
        logging.info("Running readmission model on test data...")
        try:
            readmission_results = readmission_model.main(
                test_dir,
                tune_models=tune_models,
                use_feature_selection=False
            )
            logging.info("Readmission model completed successfully.")
        except Exception as e:
            logging.error(f"Error running readmission model: {e}")
            import traceback
            logging.error(traceback.format_exc())
            readmission_results = None
    
    # Run resource model if specified
    if models in ['all', 'resource']:
        logging.info("Running resource model on test data...")
        try:
            resource_results, resource_summary = resource_model.main(
                test_dir,
                tune_models=tune_models,
                use_cross_validation=True,  # Use cross-validation to avoid R² score issues
                test_size=0.3  # Use larger test size to avoid R² score issues
            )
            logging.info("Resource model completed successfully.")
        except Exception as e:
            logging.error(f"Error running resource model: {e}")
            import traceback
            logging.error(traceback.format_exc())
            resource_results, resource_summary = None, None
    
    # Calculate total runtime
    total_time = time.time() - start_time
    logging.info(f"Test pipeline completed in {total_time:.2f} seconds.")
    
    results = {
        "readmission_results": readmission_results,
        "resource_results": resource_results,
        "resource_summary": resource_summary,
        "runtime_seconds": total_time
    }
    
    # Save results to output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save summary as CSV
        summary_df = pd.DataFrame({
            'model': ['readmission', 'resource'],
            'status': [
                'success' if readmission_results is not None else 'failed',
                'success' if resource_results is not None else 'failed'
            ],
            'runtime': [total_time, total_time]
        })
        summary_df.to_csv(output_path / 'test_summary.csv', index=False)
        logging.info(f"Results saved to {output_dir}")
        
        # Save detailed results if available
        if isinstance(readmission_results, dict) and 'metrics_df' in readmission_results:
            readmission_results['metrics_df'].to_csv(output_path / 'readmission_metrics.csv', index=False)
        
        if isinstance(resource_results, dict) and 'metrics_df' in resource_results:
            resource_results['metrics_df'].to_csv(output_path / 'resource_metrics.csv', index=False)
    
    return results

def main():
    """Main function to run the test pipeline"""
    parser = argparse.ArgumentParser(description='Run the model pipeline on test data')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base directory for data')
    parser.add_argument('--no-create-subset', action='store_false', dest='create_subset',
                        help='Skip creating a new test subset')
    parser.add_argument('--subset-size', type=float, default=0.2,
                        help='Proportion of data to include in subset (default: 0.2)')
    parser.add_argument('--tune', action='store_true',
                        help='Tune model hyperparameters')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save test results')
    parser.add_argument('--models', type=str, default='all',
                        choices=['all', 'readmission', 'resource'],
                        help='Which models to test (default: all)')
    
    args = parser.parse_args()
    
    run_test_pipeline(
        data_dir=args.data_dir,
        create_subset=args.create_subset,
        subset_size=args.subset_size,
        tune_models=args.tune,
        output_dir=args.output_dir,
        models=args.models
    )

if __name__ == "__main__":
    main()
