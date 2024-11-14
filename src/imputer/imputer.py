"""
This module contains the main Imputer class used to handle missing values in a dataset.
"""

import pandas as pd
import openai

class Imputer:
    """
    A class used to identify and impute missing values in a DataFrame.
    """

    def __init__(self, api_key: str):
        """
        Initializes the Imputer instance and sets up OpenAI API key.

        Parameters:
        api_key (str): Your OpenAI API key for accessing the LLM.
        """
        self.data = None  # To store the input DataFrame
        self.api_key = api_key
        openai.api_key = self.api_key

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

    def impute_with_llm(self, columns: list):
        """
        Imputes missing values in specified columns using OpenAI's LLM.

        Parameters:
        columns (list): List of column names in the DataFrame to impute missing values for.
        """
        if self.data is None:
            raise ValueError("No data available. Please fit the imputer with a DataFrame first.")
        
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

            for i, row in self.data.iterrows():
                if pd.isna(row[col]):
                    # Generate a prompt using other non-missing row values
                    context = row.dropna().to_dict()  # Get non-missing values for the row
                    prompt = f"Based on the following information, provide a value for '{col}':\n{context}"
                    
                    # Call OpenAI API to get the imputed value
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        max_tokens=10
                    )
                    imputed_value = response.choices[0].text.strip()
                    
                    # Update the DataFrame with the imputed value
                    self.data.at[i, col] = imputed_value
