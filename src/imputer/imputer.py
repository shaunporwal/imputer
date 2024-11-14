"""
This module contains the main Imputer class used to handle missing values in a dataset.
"""

import pandas as pd

class Imputer:
    """
    A class used to identify and impute missing values in a DataFrame.
    """

    def __init__(self):
        """
        Initializes the Imputer instance.
        """
        self.data = None  # To store the input DataFrame
        
    def hello(self):
        """
        Returns a greeting message.
        
        Returns:
        str: A greeting message.
        """
        return "Hello from the Imputer class!"

    def fit(self, data: pd.DataFrame):
        """
        Stores the DataFrame for processing.
        
        Parameters:
        data (pd.DataFrame): The input DataFrame containing data to be processed.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        self.data = data

    def get_rows_with_na(self) -> pd.DataFrame:
        """
        Returns rows in the DataFrame that contain missing values.

        Returns:
        pd.DataFrame: A DataFrame with rows that contain at least one NA value.
        """
        if self.data is None:
            raise ValueError("No data available. Please fit the imputer with a DataFrame first.")
        
        # Identify rows with NA values
        rows_with_na = self.data[self.data.isna().any(axis=1)]
        return rows_with_na
