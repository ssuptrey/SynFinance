"""
Test suite for column variance and data quality validation.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

DATASET_PATH = Path("output/week3_analysis_dataset.csv")
VARIANCE_RESULTS_PATH = Path("output/variance_analysis_results.json")

@pytest.fixture(scope="module")
def dataset():
    if not DATASET_PATH.exists():
        pytest.skip(f"Dataset not found")
    return pd.read_csv(DATASET_PATH)

@pytest.fixture(scope="module")
def variance_results():
    if not VARIANCE_RESULTS_PATH.exists():
        pytest.skip(f"Results not found")
    with open(VARIANCE_RESULTS_PATH, 'r') as f:
        return json.load(f)

class TestNumericalFieldVariance:
    def test_amount_has_sufficient_variance(self, variance_results):
        result = variance_results.get('Amount', {})
        assert result.get('cv', 0) > 0.5
    
    def test_amount_is_right_skewed(self, variance_results):
        result = variance_results.get('Amount', {})
        assert result.get('skewness', 0) > 0
    
    def test_amount_has_no_negatives(self, dataset):
        assert (dataset['Amount'] >= 0).all()
    
    def test_distance_has_sufficient_variance(self, variance_results):
        result = variance_results.get('Distance_from_Home', {})
        assert result.get('cv', 0) > 0.5
    
    def test_merchant_reputation_in_valid_range(self, dataset):
        assert (dataset['Merchant_Reputation'] >= 0).all()
        assert (dataset['Merchant_Reputation'] <= 1).all()
    
    def test_customer_age_is_realistic(self, dataset):
        assert (dataset['Customer_Age'] >= 18).all()
        assert (dataset['Customer_Age'] <= 80).all()

class TestCategoricalFieldDiversity:
    def test_payment_mode_has_good_diversity(self, variance_results):
        result = variance_results.get('Payment_Mode', {})
        assert result.get('entropy', 0) > 2.0
        assert result.get('unique_values', 0) >= 5
    
    def test_category_has_high_diversity(self, variance_results):
        result = variance_results.get('Category', {})
        assert result.get('entropy', 0) > 3.5
        assert result.get('unique_values', 0) >= 15
    
    def test_customer_segment_has_all_7_segments(self, variance_results):
        result = variance_results.get('Customer_Segment', {})
        assert result.get('unique_values', 0) == 7
    
    def test_city_has_good_geographic_diversity(self, variance_results):
        result = variance_results.get('City', {})
        assert result.get('unique_values', 0) >= 20
        assert result.get('entropy', 0) > 4.0

class TestDataQualityOverall:
    def test_dataset_has_expected_size(self, dataset):
        assert len(dataset) == 10000
    
    def test_dataset_has_expected_columns(self, dataset):
        assert len(dataset.columns) == 45
    
    def test_overall_missing_data_rate(self, dataset):
        total_cells = dataset.shape[0] * dataset.shape[1]
        missing_cells = dataset.isna().sum().sum()
        missing_rate = (missing_cells / total_cells) * 100
        assert missing_rate < 6  # Actual: ~5.25%, mostly from Card_Type (51%) and optional fields

pytestmark = pytest.mark.variance
