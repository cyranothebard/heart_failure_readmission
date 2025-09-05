import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import readmission_model, resource_model

class TestReadmissionModel(unittest.TestCase):
    
    def setUp(self):
        """Create temporary directory and mock data for tests"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create mock data
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'categorical': ['A', 'B', 'A', 'B', 'C']
        })
        self.X_test = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [0.6, 0.7],
            'categorical': ['A', 'B']
        })
        self.y_train = np.array([0, 1, 0, 1, 0])
        self.y_test = np.array([1, 0])
        
        # Save mock data to the temp directory
        self.X_train.to_csv(os.path.join(self.processed_dir, 'X_train.csv'), index=False)
        self.X_test.to_csv(os.path.join(self.processed_dir, 'X_test.csv'), index=False)
        pd.DataFrame({'target': self.y_train}).to_csv(os.path.join(self.processed_dir, 'y_train.csv'), index=False)
        pd.DataFrame({'target': self.y_test}).to_csv(os.path.join(self.processed_dir, 'y_test.csv'), index=False)
    
    def tearDown(self):
        """Remove temporary directory and files"""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.models.readmission_model.load_data')
    def test_preprocess_features(self, mock_load_data):
        """Test preprocessing of features with mock data"""
        # Call the preprocess function
        X_train_processed, X_test_processed = readmission_model.preprocess_features(
            self.X_train, self.X_test
        )
        
        # Check that categorical features were transformed
        self.assertNotIn('categorical', X_train_processed.columns)
        self.assertIn('categorical_B', X_train_processed.columns)
        self.assertIn('categorical_C', X_train_processed.columns)
        
        # Check that numerical features remain unchanged
        self.assertIn('feature1', X_train_processed.columns)
        self.assertIn('feature2', X_train_processed.columns)
        
        # Check that the shapes are correct
        self.assertEqual(X_train_processed.shape, (5, 4))  # 5 rows, 4 columns (2 original + 2 encoded)
        self.assertEqual(X_test_processed.shape, (2, 4))   # 2 rows, 4 columns
    
    @patch('src.models.readmission_model.LogisticRegression')
    def test_train_logistic_regression(self, mock_lr):
        """Test logistic regression model training with mock data"""
        # Setup mock
        mock_model = MagicMock()
        mock_lr.return_value = mock_model
        
        # Call the function to train a model
        model = readmission_model.train_logistic_regression(
            self.X_train, self.y_train
        )
        
        # Assert that model was trained
        mock_model.fit.assert_called_once()
        args, kwargs = mock_model.fit.call_args
        # Check that fit was called with the right data
        self.assertTrue((args[0] == self.X_train).all().all())
        self.assertTrue((args[1] == self.y_train).all())
    
    @patch('src.models.readmission_model.evaluate_model')
    @patch('src.models.readmission_model.train_logistic_regression')
    @patch('src.models.readmission_model.load_data')
    @patch('src.models.readmission_model.plot_feature_importance')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_main_function_basic_flow(self, mock_close, mock_savefig, mock_figure, 
                                     mock_plot_fi, mock_load_data, mock_train, mock_evaluate):
        """Test the basic flow of the main function with mocked dependencies"""
        # Setup mocks
        mock_load_data.return_value = (self.X_train, self.X_test, self.y_train, self.y_test)
        mock_model = MagicMock()
        mock_train.return_value = mock_model
        mock_evaluate.return_value = {
            'model_name': 'Logistic Regression', 
            'roc_auc': 0.8,
            'avg_precision': 0.75,
            'f1_score': 0.78,
            'precision': 0.8,
            'recall': 0.76,
            'accuracy': 0.82
        }
        
        # Mock feature importance specifically to return fixed values
        mock_fi_plot = MagicMock()
        mock_fi_df = pd.DataFrame({'Feature': ['feature1', 'feature2'], 
                                   'Importance': [0.7, 0.3]})
        mock_plot_fi.return_value = (mock_fi_plot, mock_fi_df)
        
        # Create temporary directories for outputs
        models_dir = os.path.join(self.temp_dir, 'models')
        plots_dir = os.path.join(self.temp_dir, '..', 'reports', 'figures')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Run with minimal model training to test the main flow
        with patch('src.models.readmission_model.os.makedirs'):
            with patch('src.models.readmission_model.os.path.join', return_value=self.processed_dir):
                with patch('src.models.readmission_model.plot_roc_curve'):
                    with patch('src.models.readmission_model.plot_pr_curve'):
                        with patch('src.models.readmission_model.save_model'):
                            with patch('src.models.readmission_model.pd.DataFrame.to_csv'):
                                result = readmission_model.main(
                                    self.temp_dir, 
                                    tune_models=False,
                                    use_feature_selection=False
                                )
        
        # Assert that load_data was called
        mock_load_data.assert_called_once()
        
        # Assert that at least one model was trained (Logistic Regression)
        mock_train.assert_called()


class TestResourceModel(unittest.TestCase):
    
    def setUp(self):
        """Create temporary directory and mock data for tests"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create mock data
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'categorical': ['A', 'B', 'A', 'B', 'C']
        })
        self.X_test = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [0.6, 0.7],
            'categorical': ['A', 'B']
        })
        self.y_train = np.array([5.1, 6.2, 7.3, 8.4, 9.5])
        self.y_test = np.array([10.1, 11.2])
        
        # Save mock data for the regression task
        self.X_train.to_csv(os.path.join(self.processed_dir, 'X_train.csv'), index=False)
        self.X_test.to_csv(os.path.join(self.processed_dir, 'X_test.csv'), index=False)
        pd.DataFrame({'total_los_days': self.y_train}).to_csv(
            os.path.join(self.processed_dir, 'total_los_days_train.csv'), index=False
        )
        pd.DataFrame({'total_los_days': self.y_test}).to_csv(
            os.path.join(self.processed_dir, 'total_los_days_test.csv'), index=False
        )
        
        # Create a mock processed data file with LOS columns
        self.processed_data = pd.DataFrame({
            'hadm_id': [101, 102, 103, 104, 105, 106, 107],
            'admittime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', 
                                         '2023-01-04', '2023-01-05', '2023-01-06', 
                                         '2023-01-07']),
            'dischtime': pd.to_datetime(['2023-01-05', '2023-01-08', '2023-01-06', 
                                         '2023-01-10', '2023-01-08', '2023-01-12', 
                                         '2023-01-10']),
            'hypertension': [1, 0, 1, 1, 0, 1, 0],
            'diabetes': [0, 1, 1, 0, 0, 1, 1],
            'kidney_disease': [0, 0, 1, 0, 0, 1, 0],
            'copd': [0, 0, 0, 1, 0, 0, 0],
            'atrial_fibrillation': [1, 0, 0, 0, 0, 1, 0],
            'coronary_artery_disease': [0, 1, 0, 0, 0, 1, 0],
            'obesity': [0, 1, 0, 0, 1, 0, 0],
            'anemia': [0, 0, 1, 0, 0, 0, 1],
        })
        self.processed_data.to_csv(os.path.join(self.processed_dir, 'hf_data_processed.csv'), index=False)
    
    def tearDown(self):
        """Remove temporary directory and files"""
        shutil.rmtree(self.temp_dir)
    
    def test_calculate_length_of_stay(self):
        """Test the calculation of length of stay metrics"""
        # Call the function
        result = resource_model.calculate_length_of_stay(self.processed_dir)
        
        # Check that LOS columns were calculated correctly
        self.assertIn('total_los_days', result.columns)
        self.assertIn('icu_los_days', result.columns)
        self.assertIn('non_icu_los_days', result.columns)
        
        # Check some calculations
        self.assertEqual(result['total_los_days'].iloc[0], 4.0)  # 4 days between Jan 1 and Jan 5
        
        # Verify that complex cases are handled
        # Patient with multiple comorbidities should have higher ICU fraction
        high_comorbidity_idx = 2  # Patient with 3 comorbidities
        low_comorbidity_idx = 4   # Patient with 1 comorbidity
        
        self.assertGreater(
            result['icu_los_days'].iloc[high_comorbidity_idx] / result['total_los_days'].iloc[high_comorbidity_idx],
            result['icu_los_days'].iloc[low_comorbidity_idx] / result['total_los_days'].iloc[low_comorbidity_idx]
        )
    
    @patch('src.models.resource_model.LinearRegression')
    def test_train_linear_regression(self, mock_lr):
        """Test linear regression model training with mock data"""
        # Setup mock
        mock_model = MagicMock()
        mock_lr.return_value = mock_model
        
        # Call the function
        model = resource_model.train_linear_regression(self.X_train, self.y_train)
        
        # Assert model was trained
        mock_model.fit.assert_called_once()
        args, kwargs = mock_model.fit.call_args
        # Check that fit was called with the right data
        self.assertTrue((args[0] == self.X_train).all().all())
        self.assertTrue((args[1] == self.y_train).all())
    
    @patch('src.models.resource_model.load_data')
    def test_preprocess_features(self, mock_load_data):
        """Test preprocessing of features for resource model"""
        # Call the preprocess function
        X_train_processed, X_test_processed = resource_model.preprocess_features(
            self.X_train, self.X_test
        )
        
        # Check that categorical features were transformed
        self.assertNotIn('categorical', X_train_processed.columns)
        self.assertIn('categorical_B', X_train_processed.columns)
        self.assertIn('categorical_C', X_train_processed.columns)
        
        # Check that numerical features remain unchanged
        self.assertIn('feature1', X_train_processed.columns)
        self.assertIn('feature2', X_train_processed.columns)
    
    def test_data_type_handling(self):
        """Test handling of different data types"""
        
        # Create data with mixed types
        X_train_mixed = self.X_train.copy()
        X_train_mixed['feature1'] = X_train_mixed['feature1'].astype(float)
        X_train_mixed['feature2'] = X_train_mixed['feature2'].astype(str)  # Convert to string
        X_train_mixed['categorical'] = X_train_mixed['categorical'].astype('category')  # Pandas category
        
        # Create data with missing values
        X_train_null = self.X_train.copy()
        X_train_null.loc[0, 'feature1'] = None
        X_train_null.loc[1, 'categorical'] = None
        
        # Test preprocessing with mixed types
        try:
            X_processed, _ = resource_model.preprocess_features(
                X_train_mixed, self.X_test.copy()
            )
            # Check if conversion happened
            self.assertTrue(np.issubdtype(X_processed['feature1'].dtype, np.number))
        except Exception as e:
            self.fail(f"Processing mixed data types failed: {str(e)}")
            
        # Test preprocessing with nulls
        try:
            X_processed, _ = resource_model.preprocess_features(
                X_train_null, self.X_test.copy()
            )
            # Check if nulls were handled
            self.assertFalse(X_processed.isnull().any().any())
        except Exception as e:
            self.fail(f"Processing data with nulls failed: {str(e)}")
    
    @patch('src.models.resource_model.load_data')
    @patch('src.models.resource_model.train_linear_regression')
    @patch('src.models.resource_model.evaluate_model')
    @patch('src.models.resource_model.plot_feature_importance')
    @patch('src.models.resource_model.plot_quantile_predictions')
    @patch('src.models.resource_model.plot_actual_vs_predicted')
    @patch('src.models.resource_model.plot_residuals')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('sklearn.metrics.r2_score')  # Add patch for r2_score
    def test_main_function_basic_flow(self, mock_r2_score, mock_close, mock_savefig, mock_figure,
                                     mock_plot_residuals, mock_plot_actual_vs_pred, 
                                     mock_plot_quant, mock_plot_fi, 
                                     mock_evaluate, mock_train, mock_load_data):
        """Test the basic flow of the resource model main function with mocked dependencies"""
        # Setup mocks
        mock_load_data.return_value = (
            self.X_train, self.X_test, self.y_train, self.y_test, 
            self.X_train.columns.tolist(), 'total_los_days'
        )
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([9.0, 10.0])
        mock_train.return_value = mock_model
        mock_evaluate.return_value = {
            'model_name': 'Linear Regression', 
            'r2': 0.75, 
            'rmse': 1.2,
            'mae': 0.9,
            'mape': 10.5
        }
        
        # Mock r2_score to prevent warnings
        mock_r2_score.return_value = 0.75
        
        # Mock plot returns
        mock_plot_quant.return_value = MagicMock()
        mock_plot_actual_vs_pred.return_value = MagicMock()
        mock_plot_residuals.return_value = MagicMock()
        
        # Mock feature importance specifically to return fixed values
        mock_fi_plot = MagicMock()
        mock_fi_df = pd.DataFrame({'Feature': ['feature1', 'feature2'], 
                                   'Importance': [0.7, 0.3]})
        mock_plot_fi.return_value = (mock_fi_plot, mock_fi_df)
        
        # Create temporary directories for outputs
        models_dir = os.path.join(self.temp_dir, 'models')
        plots_dir = os.path.join(self.temp_dir, '..', 'reports', 'figures')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Run with minimal model training
        with patch('src.models.resource_model.os.makedirs'):
            with patch('src.models.resource_model.os.path.join', return_value=self.processed_dir):
                with patch('src.models.resource_model.save_model'):
                    with patch('src.models.resource_model.pd.DataFrame.to_csv'):
                        with patch('src.models.resource_model.predict_resource_needs',
                                  return_value=(self.X_test, {'mean_prediction': 7.5})):
                            result, summary = resource_model.main(
                                self.temp_dir, 
                                tune_models=False
                            )
        
        # Assert that load_data was called
        mock_load_data.assert_called_once()
        
        # Assert that at least linear regression was trained
        mock_train.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_functions(self, mock_savefig):
        """Test that plotting functions run without errors"""
        
        # Set up minimal test data
        y_true = np.array([5.1, 6.2, 7.3, 8.4])
        y_pred = np.array([5.5, 6.0, 7.0, 8.0])
        feature_names = ['feature1', 'feature2']
        
        # Test residual plots
        fig = resource_model.plot_residuals(y_true, y_pred)
        self.assertIsNotNone(fig)
        
        # Test actual vs predicted
        fig = resource_model.plot_actual_vs_predicted(y_true, y_pred)
        self.assertIsNotNone(fig)
        
        # Test feature importance
        importance = np.array([0.7, 0.3])
        fig, _ = resource_model.plot_feature_importance(importance, feature_names)
        self.assertIsNotNone(fig)
        
        # Check that savefig was called at least once
        mock_savefig.assert_called()

    def test_with_realistic_data(self):
        """Test model with realistic data from factory"""
        from .test_data_factory import TestDataFactory
        
        # Generate realistic data
        X_train, X_test, y_train, y_test = TestDataFactory.create_regression_data(
            n_samples=50,  # Larger sample for more realistic behavior
            n_features=8   # More features like real data
        )
        
        # Preprocess the data
        X_train_processed, X_test_processed = resource_model.preprocess_features(
            X_train, X_test
        )
        
        # Train a model (real training, not mocked)
        model = resource_model.train_linear_regression(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Check that model produces reasonable predictions
        self.assertTrue(np.all(y_pred > 0))  # LOS should be positive
        
        # Evaluate model
        metrics = resource_model.evaluate_model(model, X_test_processed, y_test)
        
        # Check that metrics are calculated
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
    
    def test_edge_cases(self):
        """Test edge cases like empty data and extreme values"""
        
        # Empty data
        X_empty = pd.DataFrame(columns=self.X_train.columns)
        y_empty = np.array([])
        
        # Test empty data handling
        with self.assertRaises(ValueError):  # Should raise ValueError
            resource_model.train_linear_regression(X_empty, y_empty)
        
        # Extreme values
        X_extreme = self.X_train.copy()
        X_extreme['feature1'] = X_extreme['feature1'] * 1000000  # Very large values
        y_extreme = self.y_train * 1000000
        
        # Test training with extreme values
        model = resource_model.train_linear_regression(X_extreme, y_extreme)
        self.assertIsNotNone(model)
        
        # Test prediction with extreme values
        y_pred = model.predict(X_extreme)
        self.assertTrue(np.all(np.isfinite(y_pred)))  # No infinities or NaNs


if __name__ == '__main__':
    unittest.main()

# Script to create real data sample for testing
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models import resource_model

def create_real_data_sample():
    """Create a small sample of real data for testing"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    
    try:
        # Load real data
        real_data = pd.read_csv(os.path.join(data_dir, 'hf_data_processed.csv'))
        
        # Calculate length of stay if needed
        if 'total_los_days' not in real_data.columns:
            real_data = resource_model.calculate_length_of_stay(data_dir)
        
        # Take a random sample
        sample = real_data.sample(n=min(50, len(real_data)), random_state=42)
        
        # Save the sample
        sample_path = os.path.join(test_data_dir, 'real_data_sample.csv')
        sample.to_csv(sample_path, index=False)
        print(f"Created real data sample with {len(sample)} records at {sample_path}")
        
        # Print column information
        print("\nColumn data types:")
        for col, dtype in sample.dtypes.items():
            print(f"{col}: {dtype}")
        
        # Check for null values
        nulls = sample.isnull().sum()
        print("\nNull values:")
        print(nulls[nulls > 0])
        
    except Exception as e:
        print(f"Error creating real data sample: {str(e)}")

if __name__ == "__main__":
    create_real_data_sample()