"""
Unit tests for Heart Failure Readmission Prediction - Data Loader
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import MIMICDataLoader, MIMICDataProcessor


class TestMIMICDataLoader:
    """Test cases for MIMICDataLoader class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'database_path': 'test_data/mimic-iv',
            'version': '2.0',
            'chunk_size': 1000
        }
    
    @pytest.fixture
    def mock_data_loader(self, mock_config):
        """Mock MIMICDataLoader instance"""
        with patch('data.data_loader.MIMICDataLoader._create_connection_pool'):
            loader = MIMICDataLoader(mock_config)
            loader.connection_pool = Mock()
            return loader
    
    def test_init(self, mock_config):
        """Test MIMICDataLoader initialization"""
        with patch('data.data_loader.MIMICDataLoader._create_connection_pool'):
            loader = MIMICDataLoader(mock_config)
            assert loader.config == mock_config
            assert loader.connection_pool is not None
    
    def test_load_patient_data(self, mock_data_loader):
        """Test loading patient data from MIMIC-IV"""
        # Mock database response
        mock_data = pd.DataFrame({
            'subject_id': [1, 2, 3],
            'gender': ['M', 'F', 'M'],
            'dob': ['1980-01-01', '1975-05-15', '1990-12-10'],
            'hadm_id': [101, 102, 103],
            'admittime': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'dischtime': ['2020-01-05', '2020-02-05', '2020-03-05']
        })
        
        mock_data_loader.connection_pool.execute_query.return_value = mock_data
        
        # Test loading patient data
        patient_ids = [1, 2, 3]
        result = mock_data_loader.load_patient_data(patient_ids)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'subject_id' in result.columns
        assert 'gender' in result.columns
        mock_data_loader.connection_pool.execute_query.assert_called_once()
    
    def test_load_patient_data_empty(self, mock_data_loader):
        """Test loading patient data with empty result"""
        mock_data_loader.connection_pool.execute_query.return_value = pd.DataFrame()
        
        result = mock_data_loader.load_patient_data([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_load_patient_data_invalid_ids(self, mock_data_loader):
        """Test loading patient data with invalid IDs"""
        mock_data_loader.connection_pool.execute_query.return_value = pd.DataFrame()
        
        result = mock_data_loader.load_patient_data([999, 1000])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestMIMICDataProcessor:
    """Test cases for MIMICDataProcessor class"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'database_path': 'test_data/mimic-iv',
            'version': '2.0',
            'chunk_size': 1000
        }
    
    @pytest.fixture
    def mock_processor(self, mock_config):
        """Mock MIMICDataProcessor instance"""
        with patch('data.data_loader.MIMICDataProcessor._create_connection_pool'):
            processor = MIMICDataProcessor(mock_config)
            processor.connection_pool = Mock()
            return processor
    
    def test_init(self, mock_config):
        """Test MIMICDataProcessor initialization"""
        with patch('data.data_loader.MIMICDataProcessor._create_connection_pool'):
            processor = MIMICDataProcessor(mock_config)
            assert processor.config == mock_config
            assert processor.connection_pool is not None
    
    def test_process_patient_data(self, mock_processor):
        """Test processing patient data"""
        # Mock patient data
        patient_data = {
            'subject_id': 1,
            'gender': 'M',
            'age': 65,
            'admittime': '2020-01-01',
            'dischtime': '2020-01-05'
        }
        
        # Mock database responses
        mock_processor.connection_pool.execute_query.side_effect = [
            pd.DataFrame({'icd_code': ['I50', 'E11', 'N18']}),  # ICD codes
            pd.DataFrame({
                'charttime': ['2020-01-01', '2020-01-02'],
                'heartrate': [80, 85],
                'sysbp': [120, 125]
            }),  # Vital signs
            pd.DataFrame({
                'charttime': ['2020-01-01', '2020-01-02'],
                'itemid': [51265, 51265],
                'valuenum': [7.2, 7.4]
            })  # Lab values
        ]
        
        result = mock_processor.process_patient_data(patient_data)
        
        # Assertions
        assert isinstance(result, dict)
        assert 'demographics' in result
        assert 'icd_codes' in result
        assert 'vital_signs' in result
        assert 'lab_values' in result
    
    def test_process_patient_data_missing_data(self, mock_processor):
        """Test processing patient data with missing information"""
        patient_data = {
            'subject_id': 1,
            'gender': 'M',
            'age': 65
        }
        
        # Mock empty database responses
        mock_processor.connection_pool.execute_query.return_value = pd.DataFrame()
        
        result = mock_processor.process_patient_data(patient_data)
        
        # Assertions
        assert isinstance(result, dict)
        assert 'demographics' in result
        assert 'icd_codes' in result
        assert 'vital_signs' in result
        assert 'lab_values' in result
    
    def test_extract_demographics(self, mock_processor):
        """Test extracting demographic information"""
        patient_data = {
            'subject_id': 1,
            'gender': 'M',
            'age': 65,
            'race': 'White',
            'insurance': 'Medicare'
        }
        
        result = mock_processor._extract_demographics(patient_data)
        
        assert result['gender'] == 'M'
        assert result['age'] == 65
        assert result['race'] == 'White'
        assert result['insurance'] == 'Medicare'
    
    def test_extract_demographics_missing_fields(self, mock_processor):
        """Test extracting demographics with missing fields"""
        patient_data = {
            'subject_id': 1,
            'gender': 'M'
        }
        
        result = mock_processor._extract_demographics(patient_data)
        
        assert result['gender'] == 'M'
        assert result['age'] is None
        assert result['race'] is None
        assert result['insurance'] is None


class TestDataValidation:
    """Test cases for data validation functions"""
    
    def test_validate_patient_data(self):
        """Test patient data validation"""
        # Valid patient data
        valid_data = {
            'subject_id': 1,
            'gender': 'M',
            'age': 65,
            'admittime': '2020-01-01',
            'dischtime': '2020-01-05'
        }
        
        # Test validation
        assert self._validate_patient_data(valid_data) == True
        
        # Invalid patient data
        invalid_data = {
            'subject_id': None,
            'gender': 'X',
            'age': -5
        }
        
        assert self._validate_patient_data(invalid_data) == False
    
    def test_validate_icd_codes(self):
        """Test ICD code validation"""
        # Valid ICD codes
        valid_codes = ['I50', 'E11', 'N18', 'J44']
        assert self._validate_icd_codes(valid_codes) == True
        
        # Invalid ICD codes
        invalid_codes = ['INVALID', '123', 'I50.999']
        assert self._validate_icd_codes(invalid_codes) == False
    
    def test_validate_vital_signs(self):
        """Test vital signs validation"""
        # Valid vital signs
        valid_vitals = {
            'heartrate': 80,
            'sysbp': 120,
            'diasbp': 80,
            'tempc': 37.0
        }
        assert self._validate_vital_signs(valid_vitals) == True
        
        # Invalid vital signs
        invalid_vitals = {
            'heartrate': -10,
            'sysbp': 300,
            'diasbp': 50
        }
        assert self._validate_vital_signs(invalid_vitals) == False
    
    def _validate_patient_data(self, data):
        """Helper function to validate patient data"""
        required_fields = ['subject_id', 'gender']
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        
        # Check gender validity
        if data['gender'] not in ['M', 'F']:
            return False
        
        # Check age validity
        if 'age' in data and data['age'] is not None:
            if data['age'] < 0 or data['age'] > 150:
                return False
        
        return True
    
    def _validate_icd_codes(self, codes):
        """Helper function to validate ICD codes"""
        if not codes:
            return False
        
        # Basic ICD-10 format validation
        for code in codes:
            if not isinstance(code, str) or len(code) < 3:
                return False
            
            # Check if code starts with letter followed by digits
            if not (code[0].isalpha() and code[1:].replace('.', '').isdigit()):
                return False
        
        return True
    
    def _validate_vital_signs(self, vitals):
        """Helper function to validate vital signs"""
        if not vitals:
            return False
        
        # Heart rate validation
        if 'heartrate' in vitals:
            hr = vitals['heartrate']
            if hr < 30 or hr > 250:
                return False
        
        # Blood pressure validation
        if 'sysbp' in vitals:
            sysbp = vitals['sysbp']
            if sysbp < 50 or sysbp > 250:
                return False
        
        if 'diasbp' in vitals:
            diasbp = vitals['diasbp']
            if diasbp < 30 or diasbp > 150:
                return False
        
        # Temperature validation
        if 'tempc' in vitals:
            temp = vitals['tempc']
            if temp < 30 or temp > 45:
                return False
        
        return True


class TestDataProcessing:
    """Test cases for data processing functions"""
    
    def test_clean_patient_data(self):
        """Test cleaning patient data"""
        # Raw patient data with issues
        raw_data = pd.DataFrame({
            'subject_id': [1, 2, 3, 4],
            'gender': ['M', 'F', 'M', 'X'],
            'age': [65, 70, -5, 80],
            'race': ['White', 'Black', 'Asian', None]
        })
        
        cleaned_data = self._clean_patient_data(raw_data)
        
        # Assertions
        assert len(cleaned_data) == 3  # One row should be removed
        assert all(cleaned_data['gender'].isin(['M', 'F']))
        assert all(cleaned_data['age'] > 0)
        assert 'race' in cleaned_data.columns
    
    def test_handle_missing_data(self):
        """Test handling missing data"""
        # Data with missing values
        data_with_missing = pd.DataFrame({
            'heartrate': [80, None, 85, 90],
            'sysbp': [120, 125, None, 130],
            'tempc': [37.0, 36.5, 37.2, None]
        })
        
        processed_data = self._handle_missing_data(data_with_missing)
        
        # Assertions
        assert not processed_data.isnull().any().any()
        assert len(processed_data) == 4
    
    def test_normalize_data(self):
        """Test data normalization"""
        # Raw data
        raw_data = pd.DataFrame({
            'heartrate': [60, 80, 100, 120],
            'sysbp': [100, 120, 140, 160]
        })
        
        normalized_data = self._normalize_data(raw_data)
        
        # Assertions
        assert normalized_data['heartrate'].mean() == pytest.approx(0, abs=0.1)
        assert normalized_data['heartrate'].std() == pytest.approx(1, abs=0.1)
        assert normalized_data['sysbp'].mean() == pytest.approx(0, abs=0.1)
        assert normalized_data['sysbp'].std() == pytest.approx(1, abs=0.1)
    
    def _clean_patient_data(self, data):
        """Helper function to clean patient data"""
        # Remove invalid gender values
        data = data[data['gender'].isin(['M', 'F'])]
        
        # Remove invalid age values
        data = data[(data['age'] > 0) & (data['age'] <= 150)]
        
        return data
    
    def _handle_missing_data(self, data):
        """Helper function to handle missing data"""
        # Forward fill for time series data
        data = data.fillna(method='ffill')
        
        # Backward fill for remaining missing values
        data = data.fillna(method='bfill')
        
        # Mean imputation for remaining missing values
        data = data.fillna(data.mean())
        
        return data
    
    def _normalize_data(self, data):
        """Helper function to normalize data"""
        return (data - data.mean()) / data.std()


if __name__ == '__main__':
    pytest.main([__file__])
