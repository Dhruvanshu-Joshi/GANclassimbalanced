import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_dataset(file_path):
    """
    Load dataset from CSV, TXT, or DAT formats.
    Assumes the last column is the target variable.
    """
    # Load data based on file extension
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".txt") or file_path.endswith(".dat"):
        df = pd.read_csv(file_path, delimiter="\t")  # Adjust delimiter if needed
    else:
        raise ValueError("Unsupported file format. Use .csv, .txt, or .dat")

    # Ensure all columns except the last are numeric
    for col in df.columns[:-1]:  # Iterate over feature columns
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])  # Encode categorical columns

    # Ensure target column is numeric
    target_col = df.columns[-1]
    if df[target_col].dtype == 'object':
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    # Convert all data to float
    df = df.astype(float)

    # Split features and labels
    X = df.iloc[:, :-1].values  # Features (all columns except last)
    y = df.iloc[:, -1].values   # Target (last column)

    return X, y

def get_features_labels(df, target_column, test_size=0.2):
    """
    Split dataset into features and labels.
    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): Column name for labels.
        test_size (float): Test set proportion (default: 0.2).
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    features = df.drop(columns=[target_column])
    labels = df[target_column]
    return train_test_split(features, labels, test_size=test_size, random_state=10)

def get_real_data_for_GAN(X_train, y_train, target_values):
    """
    Extract real data samples from training set for specific target values.
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        target_values (list): List of target values to extract.
    Returns:
        dict: A dictionary with target values as keys and (X_real, y_real) as values.
    """
    real_data = {}
    for target in target_values:
        mask = y_train == target
        X_real = X_train[mask]
        y_real = np.full(X_real.shape[0], target)
        real_data[target] = (X_real, y_real)
    return real_data