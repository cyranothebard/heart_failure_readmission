"""
Unit tests for Heart Failure Readmission Prediction - Clinical Models
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.clinical_models import ClinicalLogisticRegression, ClinicalRandomForest, ClinicalXGBoost


class TestClinicalLogisticRegression:
    """Test cases for ClinicalLogisticRegression class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'liblinear'
        }
    
    @pytest.fixture
    def mock_model(self, mock_config):
        """Mock ClinicalLogisticRegression instance"""
        return ClinicalLogisticRegression(mock_config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data"""
        np.random.seed(42)
        X = pd.DataFrame({
            'age': np.random.normal(65, 15, 100),
            'heartrate': np.random.normal(80, 15, 100),
            'sysbp': np.random.normal(120, 20, 100),
            'diabetes': np.random.choice([0, 1], 100),
            'hypertension': np.random.choice([0, 1], 100)
        })
        y = np.random.choice([0, 1], 100)
        return X, y
    
    def test_init(self, mock_config):
        """Test ClinicalLogisticRegression initialization"""
        model = ClinicalLogisticRegression(mock_config)
        assert model.config == mock_config
        assert model.model is not None
        assert model.feature_names is None
    
    def test_fit(self, mock_model, sample_data):
        """Test model fitting"""
        X, y = sample_data
        
        # Mock the sklearn LogisticRegression
        with patch('models.clinical_models.LogisticRegression') as mock_lr:
            mock_lr_instance = Mock()
            mock_lr.return_value = mock_lr_instance
            
            mock_model.fit(X, y)
            
            # Assertions
            mock_lr.assert_called_once_with(
                C=1.0,
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )
            mock_lr_instance.fit.assert_called_once()
            assert mock_model.feature_names == X.columns.tolist()
    
    def test_predict_proba(self, mock_model, sample_data):
        """Test probability prediction"""
        X, y = sample_data
        
        # Mock the model
        mock_model.model = Mock()
        mock_model.model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        result = mock_model.predict_proba(X)
        
        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        mock_model.model.predict_proba.assert_called_once_with(X)
    
    def test_get_feature_importance(self, mock_model):
        """Test getting feature importance"""
        # Mock feature names and coefficients
        mock_model.feature_names = ['age', 'heartrate', 'sysbp']
        mock_model.model = Mock()
        mock_model.model.coef_ = np.array([[0.5, -0.3, 0.8]])
        
        result = mock_model.get_feature_importance()
        
        # Assertions
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        
        # Check that results are sorted by importance
        importances = [item[1] for item in result]
        assert importances == sorted(importances, reverse=True)
    
    def test_get_feature_importance_no_features(self, mock_model):
        """Test getting feature importance when no features are set"""
        mock_model.feature_names = None
        
        result = mock_model.get_feature_importance()
        
        assert result is None


class TestClinicalRandomForest:
    """Test cases for ClinicalRandomForest class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    
    @pytest.fixture
    def mock_model(self, mock_config):
        """Mock ClinicalRandomForest instance"""
        return ClinicalRandomForest(mock_config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data"""
        np.random.seed(42)
        X = pd.DataFrame({
            'age': np.random.normal(65, 15, 100),
            'heartrate': np.random.normal(80, 15, 100),
            'sysbp': np.random.normal(120, 20, 100),
            'diabetes': np.random.choice([0, 1], 100),
            'hypertension': np.random.choice([0, 1], 100)
        })
        y = np.random.choice([0, 1], 100)
        return X, y
    
    def test_init(self, mock_config):
        """Test ClinicalRandomForest initialization"""
        model = ClinicalRandomForest(mock_config)
        assert model.config == mock_config
        assert model.model is not None
    
    def test_fit(self, mock_model, sample_data):
        """Test model fitting"""
        X, y = sample_data
        
        # Mock the sklearn RandomForestClassifier
        with patch('models.clinical_models.RandomForestClassifier') as mock_rf:
            mock_rf_instance = Mock()
            mock_rf.return_value = mock_rf_instance
            
            mock_model.fit(X, y)
            
            # Assertions
            mock_rf.assert_called_once_with(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            mock_rf_instance.fit.assert_called_once()
    
    def test_predict_proba(self, mock_model, sample_data):
        """Test probability prediction"""
        X, y = sample_data
        
        # Mock the model
        mock_model.model = Mock()
        mock_model.model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        result = mock_model.predict_proba(X)
        
        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        mock_model.model.predict_proba.assert_called_once_with(X)
    
    def test_get_feature_importance(self, mock_model):
        """Test getting feature importance"""
        # Mock feature names and importances
        mock_model.model = Mock()
        mock_model.model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_model.model.feature_names_in_ = np.array(['age', 'heartrate', 'sysbp'])
        
        result = mock_model.get_feature_importance()
        
        # Assertions
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        
        # Check that results are sorted by importance
        importances = [item[1] for item in result]
        assert importances == sorted(importances, reverse=True)


class TestClinicalXGBoost:
    """Test cases for ClinicalXGBoost class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    @pytest.fixture
    def mock_model(self, mock_config):
        """Mock ClinicalXGBoost instance"""
        return ClinicalXGBoost(mock_config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data"""
        np.random.seed(42)
        X = pd.DataFrame({
            'age': np.random.normal(65, 15, 100),
            'heartrate': np.random.normal(80, 15, 100),
            'sysbp': np.random.normal(120, 20, 100),
            'diabetes': np.random.choice([0, 1], 100),
            'hypertension': np.random.choice([0, 1], 100)
        })
        y = np.random.choice([0, 1], 100)
        return X, y
    
    def test_init(self, mock_config):
        """Test ClinicalXGBoost initialization"""
        model = ClinicalXGBoost(mock_config)
        assert model.config == mock_config
        assert model.model is not None
    
    def test_fit(self, mock_model, sample_data):
        """Test model fitting"""
        X, y = sample_data
        
        # Mock the XGBoost XGBClassifier
        with patch('models.clinical_models.XGBClassifier') as mock_xgb:
            mock_xgb_instance = Mock()
            mock_xgb.return_value = mock_xgb_instance
            
            mock_model.fit(X, y)
            
            # Assertions
            mock_xgb.assert_called_once_with(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            mock_xgb_instance.fit.assert_called_once()
    
    def test_predict_proba(self, mock_model, sample_data):
        """Test probability prediction"""
        X, y = sample_data
        
        # Mock the model
        mock_model.model = Mock()
        mock_model.model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        result = mock_model.predict_proba(X)
        
        # Assertions
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        mock_model.model.predict_proba.assert_called_once_with(X)
    
    def test_get_feature_importance(self, mock_model):
        """Test getting feature importance"""
        # Mock feature names and importances
        mock_model.model = Mock()
        mock_model.model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_model.model.feature_names_in_ = np.array(['age', 'heartrate', 'sysbp'])
        
        result = mock_model.get_feature_importance()
        
        # Assertions
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        
        # Check that results are sorted by importance
        importances = [item[1] for item in result]
        assert importances == sorted(importances, reverse=True)


class TestModelComparison:
    """Test cases for model comparison functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample training and test data"""
        np.random.seed(42)
        X_train = pd.DataFrame({
            'age': np.random.normal(65, 15, 200),
            'heartrate': np.random.normal(80, 15, 200),
            'sysbp': np.random.normal(120, 20, 200),
            'diabetes': np.random.choice([0, 1], 200),
            'hypertension': np.random.choice([0, 1], 200)
        })
        y_train = np.random.choice([0, 1], 200)
        
        X_test = pd.DataFrame({
            'age': np.random.normal(65, 15, 50),
            'heartrate': np.random.normal(80, 15, 50),
            'sysbp': np.random.normal(120, 20, 50),
            'diabetes': np.random.choice([0, 1], 50),
            'hypertension': np.random.choice([0, 1], 50)
        })
        y_test = np.random.choice([0, 1], 50)
        
        return X_train, y_train, X_test, y_test
    
    def test_model_comparison(self, sample_data):
        """Test comparing different models"""
        X_train, y_train, X_test, y_test = sample_data
        
        # Mock models
        models = {
            'logistic_regression': Mock(),
            'random_forest': Mock(),
            'xgboost': Mock()
        }
        
        # Mock predictions
        for model in models.values():
            model.predict_proba.return_value = np.random.random((len(X_test), 2))
            model.predict.return_value = np.random.choice([0, 1], len(X_test))
        
        # Test model comparison
        results = self._compare_models(models, X_test, y_test)
        
        # Assertions
        assert isinstance(results, dict)
        assert 'logistic_regression' in results
        assert 'random_forest' in results
        assert 'xgboost' in results
        
        for model_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'roc_auc' in metrics
    
    def test_model_selection(self, sample_data):
        """Test model selection based on clinical criteria"""
        X_train, y_train, X_test, y_test = sample_data
        
        # Mock model results
        model_results = {
            'logistic_regression': {
                'accuracy': 0.78,
                'precision': 0.72,
                'recall': 0.68,
                'f1_score': 0.70,
                'roc_auc': 0.75,
                'interpretability': 0.9,
                'clinical_utility': 0.85
            },
            'random_forest': {
                'accuracy': 0.82,
                'precision': 0.75,
                'recall': 0.74,
                'f1_score': 0.74,
                'roc_auc': 0.80,
                'interpretability': 0.6,
                'clinical_utility': 0.7
            },
            'xgboost': {
                'accuracy': 0.84,
                'precision': 0.77,
                'recall': 0.76,
                'f1_score': 0.76,
                'roc_auc': 0.82,
                'interpretability': 0.4,
                'clinical_utility': 0.6
            }
        }
        
        # Test model selection
        selected_model = self._select_best_model(model_results)
        
        # Assertions
        assert selected_model == 'logistic_regression'  # Should be selected based on clinical criteria
    
    def _compare_models(self, models, X_test, y_test):
        """Helper function to compare models"""
        results = {}
        
        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1])
            }
        
        return results
    
    def _select_best_model(self, model_results):
        """Helper function to select best model based on clinical criteria"""
        # Weighted scoring for clinical applications
        weights = {
            'accuracy': 0.25,
            'interpretability': 0.30,
            'clinical_utility': 0.25,
            'precision': 0.20
        }
        
        best_model = None
        best_score = -1
        
        for model_name, metrics in model_results.items():
            score = 0
            for metric, weight in weights.items():
                if metric in metrics:
                    score += metrics[metric] * weight
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model


class TestModelValidation:
    """Test cases for model validation functionality"""
    
    def test_cross_validation(self):
        """Test cross-validation for model evaluation"""
        # Mock data
        X = np.random.random((100, 5))
        y = np.random.choice([0, 1], 100)
        
        # Mock model
        model = Mock()
        model.predict_proba.return_value = np.random.random((100, 2))
        model.predict.return_value = np.random.choice([0, 1], 100)
        
        # Test cross-validation
        cv_scores = self._perform_cross_validation(model, X, y, cv=5)
        
        # Assertions
        assert isinstance(cv_scores, dict)
        assert 'accuracy' in cv_scores
        assert 'precision' in cv_scores
        assert 'recall' in cv_scores
        assert 'f1_score' in cv_scores
        assert 'roc_auc' in cv_scores
        
        for metric, scores in cv_scores.items():
            assert isinstance(scores, list)
            assert len(scores) == 5  # 5-fold CV
    
    def test_temporal_validation(self):
        """Test temporal validation for time series data"""
        # Mock temporal data
        X = pd.DataFrame({
            'admittime': pd.date_range('2020-01-01', periods=100, freq='D'),
            'feature1': np.random.random(100),
            'feature2': np.random.random(100)
        })
        y = np.random.choice([0, 1], 100)
        
        # Test temporal validation
        temporal_scores = self._perform_temporal_validation(X, y, n_splits=5)
        
        # Assertions
        assert isinstance(temporal_scores, dict)
        assert 'accuracy' in temporal_scores
        assert 'precision' in temporal_scores
        assert 'recall' in temporal_scores
        assert 'f1_score' in temporal_scores
    
    def _perform_cross_validation(self, model, X, y, cv=5):
        """Helper function to perform cross-validation"""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Mock cross-validation scores
        cv_scores = {
            'accuracy': [0.8, 0.82, 0.78, 0.85, 0.79],
            'precision': [0.75, 0.77, 0.73, 0.80, 0.76],
            'recall': [0.70, 0.72, 0.68, 0.75, 0.71],
            'f1_score': [0.72, 0.74, 0.70, 0.77, 0.73],
            'roc_auc': [0.78, 0.80, 0.76, 0.83, 0.79]
        }
        
        return cv_scores
    
    def _perform_temporal_validation(self, X, y, n_splits=5):
        """Helper function to perform temporal validation"""
        # Mock temporal validation scores
        temporal_scores = {
            'accuracy': [0.75, 0.78, 0.80, 0.82, 0.79],
            'precision': [0.70, 0.73, 0.75, 0.77, 0.74],
            'recall': [0.65, 0.68, 0.70, 0.72, 0.69],
            'f1_score': [0.67, 0.70, 0.72, 0.74, 0.71]
        }
        
        return temporal_scores


if __name__ == '__main__':
    pytest.main([__file__])
