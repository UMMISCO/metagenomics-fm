import openml
import pandas as pd


def open_openml(dataset_name: str):
    """
    Load and prepare an OpenML dataset for use in cross-validation.

    Parameters
    ----------
    dataset_name : str
        Name in the format "openml_<id>", e.g. "openml_11".

    Returns
    -------
    X : pandas.DataFrame
        Numeric feature matrix.
    y : pandas.Series
        Target vector.
    """

    if not dataset_name.startswith("openml_"):
        raise ValueError(f"Invalid OpenML dataset name '{dataset_name}'. Expected format 'openml_<id>'")

    try:
        dataset_id = int(dataset_name.split("openml_")[-1])
    except ValueError:
        raise ValueError(f"Could not extract numeric ID from dataset name '{dataset_name}'")

    print(f"[OpenML] Loading dataset ID {dataset_id} ...")

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
    )

    # Keep only numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']
    X = X.select_dtypes(include=numerics)

    # # Check for missing values
    # n_missing = X.isna().sum().sum()
    # if n_missing > 0:
    #     print(f"⚠️ Dataset {dataset_name} contains {n_missing} missing values:")
    #     print(X.isna().sum())  # number of NaNs per column
    #     print("\nExample rows with missing values:")
    #     print(X[X.isna().any(axis=1)].head())
    # else:
    #     print(f"✅ Dataset {dataset_name} has no missing values.")

    X = X.dropna()
    y = y.loc[X.index]  # keep y aligned with remaining rows
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"  ✅ Loaded {dataset.name} (ID: {dataset_id}) | Shape: {X.shape}")

    return X, y
