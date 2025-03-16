import pandas as pd
import numpy as np
import os
import tempfile
import matplotlib

class TestDataFactory:
    """Factory class for generating test data for different model types"""
    
    @staticmethod
    def create_classification_data(n_samples=100, n_features=10, random_seed=42):
        """Create mock data for classification tests with realistic features"""
        np.random.seed(random_seed)
        
        # Create feature names similar to real data
        feature_names = [
            'age', 'heart_rate', 'sbp', 'dbp', 'sodium', 'potassium',
            'bun', 'creatinine', 'hematocrit', 'ldh'
        ][:n_features]
        
        # Generate feature data with realistic distributions
        X = pd.DataFrame()
        for i, name in enumerate(feature_names):
            if name == 'age':
                X[name] = np.random.normal(68, 15, n_samples).clip(18, 100)
            elif name == 'heart_rate':
                X[name] = np.random.normal(80, 20, n_samples).clip(40, 180)
            elif name == 'sbp':
                X[name] = np.random.normal(130, 25, n_samples).clip(80, 220)
            else:
                X[name] = np.random.normal(0, 1, n_samples)
                
        # Add categorical features
        categorical_vars = {'gender': ['M', 'F'], 
                           'diabetes': [0, 1], 
                           'hypertension': [0, 1]}
        
        for name, categories in categorical_vars.items():
            if len(X.columns) < n_features:
                X[name] = np.random.choice(categories, n_samples)
        
        # Generate binary target (readmission)
        y = np.random.binomial(1, 0.3, n_samples)
        
        # Split into train and test
        train_idx = np.random.choice(n_samples, int(n_samples * 0.8), replace=False)
        test_idx = np.array([i for i in range(n_samples) if i not in train_idx])
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def create_regression_data(n_samples=100, n_features=10, random_seed=42):
        """Create mock data for regression tests with LOS target"""
        np.random.seed(random_seed)
        
        # Create feature data 
        X_train, X_test, _, _ = TestDataFactory.create_classification_data(
            n_samples, n_features, random_seed
        )
        
        # Create LOS targets with realistic distribution (log-normal)
        y_train = np.exp(np.random.normal(1.6, 0.7, len(X_train))).clip(1, 30)
        y_test = np.exp(np.random.normal(1.6, 0.7, len(X_test))).clip(1, 30)
        
        # Make sure your y_test has at least 2 samples
        y_test = np.array([10.1, 11.2, 12.3])  # Add a third sample

        # Also update X_test to match the number of samples
        X_test = pd.DataFrame({
            'feature1': [6, 7, 8],
            'feature2': [0.6, 0.7, 0.8],
            'categorical': ['A', 'B', 'C']
        })

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def save_test_data(path, prefix="mock"):
        """Save standard test datasets to the specified directory"""
        os.makedirs(path, exist_ok=True)
        
        # Classification data
        X_train, X_test, y_train, y_test = TestDataFactory.create_classification_data()
        X_train.to_csv(os.path.join(path, f"{prefix}_X_train.csv"), index=False)
        X_test.to_csv(os.path.join(path, f"{prefix}_X_test.csv"), index=False)
        pd.DataFrame({'target': y_train}).to_csv(os.path.join(path, f"{prefix}_y_train.csv"), index=False)
        pd.DataFrame({'target': y_test}).to_csv(os.path.join(path, f"{prefix}_y_test.csv"), index=False)
        
        # Regression data
        X_los_train, X_los_test, y_los_train, y_los_test = TestDataFactory.create_regression_data()
        X_los_train.to_csv(os.path.join(path, f"{prefix}_X_los_train.csv"), index=False)
        X_los_test.to_csv(os.path.join(path, f"{prefix}_X_los_test.csv"), index=False)
        pd.DataFrame({'total_los_days': y_los_train}).to_csv(os.path.join(path, f"{prefix}_los_train.csv"), index=False)
        pd.DataFrame({'total_los_days': y_los_test}).to_csv(os.path.join(path, f"{prefix}_los_test.csv"), index=False)
        
        return path

def setUp(self):
    """Create temporary directory and mock data for tests"""
    matplotlib.use('Agg')  # Use non-interactive backend
    self.temp_dir = tempfile.mkdtemp()
    self.processed_dir = os.path.join(self.temp_dir, 'processed')
    os.makedirs(self.processed_dir, exist_ok=True)
    
    # Create mock data using the factory
    self.X_train, self.X_test, self.y_train, self.y_test = TestDataFactory.create_regression_data(
        n_samples=100, n_features=3, random_seed=42
    )
    
    # Rest of your setup code...
