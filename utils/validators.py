"""
Common validation functions for synthetic data quality checks.
"""
from typing import Dict, List, Tuple

import pandas as pd


def check_range(
        df: pd.DataFrame,
        column: str,
        low: float,
        high: float,
        threshold: float = 0.99
) -> Dict:
    """
    Check if values are within physiological range.

    Returns:
        Dictionary with check results
    """
    if column not in df.columns:
        return {'status': 'MISSING', 'message': f'{column} not found'}

    in_range = ((df[column] >= low) & (df[column] <= high)).mean()
    passed = in_range >= threshold

    return {
        'column': column,
        'range': f'[{low}, {high}]',
        'in_range_pct': f'{in_range:.1%}',
        'status': 'OK' if passed else 'WARN',
        'passed': passed
    }


def check_correlation(
        df: pd.DataFrame,
        col1: str,
        col2: str,
        expected_range: Tuple[float, float]
) -> Dict:
    """
    Check if correlation between two columns is within expected range.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return {'status': 'MISSING', 'message': 'Column(s) not found'}

    valid = df[[col1, col2]].dropna()
    if len(valid) < 100:
        return {'status': 'WARN', 'message': 'Insufficient data for correlation'}

    corr = valid[col1].corr(valid[col2])
    in_range = expected_range[0] <= corr <= expected_range[1]

    return {
        'columns': f'{col1} ↔ {col2}',
        'correlation': round(corr, 3),
        'expected': f'{expected_range[0]}..{expected_range[1]}',
        'status': 'OK' if in_range else 'WARN',
        'passed': in_range
    }


def check_prevalence(
        df: pd.DataFrame,
        column: str,
        target_range: Tuple[float, float]
) -> Dict:
    """
    Check disease prevalence is within target range.
    """
    if column not in df.columns:
        return {'status': 'MISSING', 'message': f'{column} not found'}

    prevalence = df[column].mean()
    in_range = target_range[0] <= prevalence <= target_range[1]

    return {
        'column': column,
        'prevalence': f'{prevalence:.1%}',
        'target': f'{target_range[0]:.0%}-{target_range[1]:.0%}',
        'status': 'OK' if in_range else 'WARN',
        'passed': in_range
    }


def run_validation_checks(checks: List[Dict]) -> pd.DataFrame:
    """
    Run multiple validation checks and return summary DataFrame.
    """
    results = []
    for check in checks:
        results.append(check)

    df = pd.DataFrame(results)
    return df


def print_validation_table(df: pd.DataFrame, title: str = "Validation Results"):
    """Print validation results as markdown table"""
    print(f"\n### {title}")
    print(df.to_markdown(index=False))
