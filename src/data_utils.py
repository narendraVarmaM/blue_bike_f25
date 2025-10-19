import pandas as pd
from datetime import datetime
from typing import Tuple


def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time series DataFrame into training and testing sets based on a cutoff date.

    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data.
        cutoff_date (datetime): The date used to split the data into training and testing sets.
        target_column (str): The name of the target column to separate from the features.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training target values.
            - X_test (pd.DataFrame): Testing features.
            - y_test (pd.Series): Testing target values.
    """
    # Split the data into training and testing sets based on the cutoff date
    train_data = df[df["hour"] < cutoff_date].reset_index(drop=True)
    test_data = df[df["hour"] >= cutoff_date].reset_index(drop=True)

    # Separate features (X) and target (y) for both training and testing sets
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test

