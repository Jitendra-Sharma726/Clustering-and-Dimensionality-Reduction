import pandas as pd

# --- Data Loading and Exploration Functions ---

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    
    TODO:
        - Ensure the CSV file exists at the given path.
        - Handle exceptions if file is missing or corrupt.
    """
    df = pd.read_csv(file_path)
    return df

def dataset_shape(df: pd.DataFrame) -> tuple:
    """
    Get the shape of the dataset (rows, columns).

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        tuple: Number of rows and columns.
    
    TODO:
        - Can be expanded to print dataset summary statistics.
    """
    return df.shape

def dataset_info(df: pd.DataFrame) -> pd.Series:
    """
    Get the data types of each column.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.Series: Data types of columns.
    
    TODO:
        - Can be expanded to include memory usage and null count.
    """
    return df.dtypes

def missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in the dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.Series: Count of missing values per column.
    
    TODO:
        - Implement strategies to handle missing values if present.
    """
    return df.isnull().sum()

def categorical_distribution(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Get the distribution of a categorical column.

    Args:
        df (pd.DataFrame): Input dataset.
        column (str): Column name.

    Returns:
        pd.Series: Value counts of the categorical column.
    
    TODO:
        - Can be extended to plot distributions for visual inspection.
    """
    return df[column].value_counts()


# --- Feature Engineering Functions ---

def create_income_to_spending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature 'income_to_spending' by dividing Annual Income by Spending Score.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with new column added.
    
    TODO:
        - Handle division by zero if Spending Score contains 0.
    """
    df['income_to_spending'] = df['Annual Income (k$)'] / df['Spending Score (1-100)']
    return df

def create_age_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature 'age_income_ratio' by dividing Age by Annual Income.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with new column added.
    
    TODO:
        - Handle division by zero if Annual Income contains 0.
    """
    df['age_income_ratio'] = df['Age'] / df['Annual Income (k$)']
    return df


# --- Main Execution Block ---
if __name__ == '__main__':
    file_path = 'Mall_Customers.csv' 
    df = load_dataset(file_path)

    # Dataset exploration
    print("Shape of dataset:", dataset_shape(df))
    print("\nData types:\n", dataset_info(df))
    print("\nMissing values:\n", missing_values(df))
    print("\nGender distribution:\n", categorical_distribution(df, 'Gender'))

    # Feature engineering
    df = create_income_to_spending(df)
    df = create_age_income_ratio(df)
    print("\nNew columns added:\n", df[['income_to_spending', 'age_income_ratio']].head())
