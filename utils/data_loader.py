"""
Standardized data loading functions for all 5 synthetic datasets.
"""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_baseline(data_path: Path, version: str) -> pd.DataFrame:
    """Load 01_cohort_baseline dataset"""
    filepath = data_path / f"01_cohort_baseline_{version}.csv"
    return pd.read_csv(filepath)


def load_lifestyle(data_path: Path, version: str) -> pd.DataFrame:
    """Load 02_lifestyle_history dataset"""
    filepath = data_path / f"02_lifestyle_history_{version}.csv"
    return pd.read_csv(filepath)


def load_biomarkers(data_path: Path, version: str) -> pd.DataFrame:
    """Load 03_biomarkers_history dataset"""
    filepath = data_path / f"03_biomarkers_history_{version}.csv"
    return pd.read_csv(filepath)


def load_risks(data_path: Path, version: str) -> pd.DataFrame:
    """Load 04_health_risks dataset"""
    filepath = data_path / f"04_health_risks_{version}.csv"
    return pd.read_csv(filepath)


def load_aggregated(data_path: Path, version: str) -> pd.DataFrame:
    """Load 05_aggregated_dataset_with_missing dataset"""
    filepath = data_path / f"05_aggregated_dataset_with_missing_{version}.csv"
    return pd.read_csv(filepath)


def load_all_datasets(
        data_path: Path,
        version: str,
        datasets: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple datasets at once.

    Args:
        data_path: Path to data directory
        version: Version string (e.g., 'v1.1.1')
        datasets: List of datasets to load (default: all 5)

    Returns:
        Dictionary with dataset names as keys
    """
    if datasets is None:
        datasets = ['baseline', 'lifestyle', 'biomarkers', 'risks', 'aggregated']

    loaders = {
        'baseline': load_baseline,
        'lifestyle': load_lifestyle,
        'biomarkers': load_biomarkers,
        'risks': load_risks,
        'aggregated': load_aggregated,
    }

    loaded = {}
    for name in datasets:
        if name in loaders:
            loaded[name] = loaders[name](data_path, version)
            print(f"  Loaded {name}: {loaded[name].shape}")

    return loaded


def verify_key_columns(df: pd.DataFrame, required_cols: list, df_name: str = "Dataset") -> bool:
    """
    Verify that all required columns are present.

    Returns:
        True if all columns present, False otherwise
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f" ! {df_name}: Missing columns: {missing}")
        return False
    print(f"  {df_name}: All {len(required_cols)} key columns present")
    return True
