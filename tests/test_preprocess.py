import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data import preprocess  # Import the preprocess module

class TestPreprocess(unittest.TestCase):
    
    def test_encode_categorical_features(self):
        # Create a sample DataFrame
        data = pd.DataFrame({
            'categorical_col1': ['A', 'B', 'A', 'C'],
            'categorical_col2': ['X', 'Y', 'Y', 'Z'],
            'numerical_col': [1, 2, 3, 4]
        })
        categorical_cols = ['categorical_col1', 'categorical_col2']
        
        # Encode the categorical features
        df_encoded, encoder = preprocess.encode_categorical_features(data, categorical_cols)
        
        # Assert that the returned DataFrame is not None
        self.assertIsNotNone(df_encoded)
        
        # Assert that the original categorical columns are dropped
        self.assertNotIn('categorical_col1', df_encoded.columns)
        self.assertNotIn('categorical_col2', df_encoded.columns)
        
        # Assert that the encoded columns are added
        self.assertIn('categorical_col1_B', df_encoded.columns)
        self.assertIn('categorical_col1_C', df_encoded.columns)
        self.assertIn('categorical_col2_Y', df_encoded.columns)
        self.assertIn('categorical_col2_Z', df_encoded.columns)
        
        # Assert that the shape of the encoded DataFrame is correct
        self.assertEqual(df_encoded.shape, (4, 5))  # 4 rows, 5 columns (1 numerical + 4 encoded)
        
        # Test when no categorical columns are provided
        df_encoded_none, encoder_none = preprocess.encode_categorical_features(data, [])
        self.assertIsNone(encoder_none)
        self.assertTrue(df_encoded_none.equals(data))
    
    def test_prepare_data_for_modeling(self):
        # Create a larger sample DataFrame for stratification
        data = pd.DataFrame({
            'feature1': list(range(1, 11)),
            'feature2': list(range(11, 21)),
            'is_readmission': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        target_col = 'is_readmission'
        # With 10 samples and a balanced class distribution
        test_size = 0.4  # Test set will have 4 samples
        random_state = 42

        # Prepare the data for modeling
        X_train, X_test, y_train, y_test = preprocess.prepare_data_for_modeling(
            data, target_col, test_size, random_state
        )

        # Assert that the returned objects are not None
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

        # With 10 samples, training set has 6 samples and test set has 4 samples
        self.assertEqual(X_train.shape, (6, 2))  # 6 rows, 2 features
        self.assertEqual(X_test.shape, (4, 2))   # 4 rows, 2 features
        self.assertEqual(y_train.shape, (6,))     # 6 rows
        self.assertEqual(y_test.shape, (4,))      # 4 rows

        # Assert that the target column is not in the feature DataFrames
        self.assertNotIn(target_col, X_train.columns)
        self.assertNotIn(target_col, X_test.columns)

        # For stratification, since there are 5 positive samples overall:
        # Training: 3 positives, Test: 2 positives (or vice-versa) depending on the split.
        self.assertEqual(sum(y_train) + sum(y_test), 5)

if __name__ == '__main__':
    unittest.main()