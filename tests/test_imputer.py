"""
This test file contains unit tests for the Imputer class, focusing on loading and storing
data as well as identifying rows with missing values. The tests use a real dataset from 
a specified GitHub URL to verify the functionality of the fit and get_rows_with_na methods.
"""

import pytest
import pandas as pd
from imputer import Imputer

# URL to the CSV file in your GitHub repository
DATA_URL = "https://raw.githubusercontent.com/shaunporwal/imputer/main/data/sim_medical_data.csv"

@pytest.fixture
def sample_data():
    """Fixture to load the CSV data from the GitHub repository."""
    data = pd.read_csv(DATA_URL)
    return data

def test_fit(sample_data):
    """Test that the fit method correctly stores the DataFrame."""
    imputer = Imputer(api_key=None)  # Using None as a placeholder for the API key
    imputer.fit(sample_data)
    assert imputer.data is not None
    pd.testing.assert_frame_equal(imputer.data, sample_data)

def test_get_rows_with_na(sample_data):
    """Test that get_rows_with_na returns rows with missing values."""
    imputer = Imputer(api_key=None)  # Using None as a placeholder for the API key
    imputer.fit(sample_data)
    rows_with_na = imputer.get_rows_with_na()

    # Expected rows with missing values
    expected_rows = sample_data[sample_data.isna().any(axis=1)]
    pd.testing.assert_frame_equal(rows_with_na, expected_rows)
