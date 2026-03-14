"""
Standalone utility functions for SyntheticHealthSimulator.

These functions have no class dependencies and can be imported
from any module without circular import issues.
"""

from typing import Union, Dict, Tuple

import numpy as np
from scipy.stats import truncnorm


def generate_truncnorm(
        rng: np.random.Generator,
        loc: Union[float, np.ndarray],
        scale: float,
        low: float,
        high: float,
        size: int,
) -> np.ndarray:
    """
    Generate values from a truncated normal distribution.

    Args:
        rng: NumPy random number generator
        loc: Mean of the distribution
        scale: Standard deviation
        low: Lower bound
        high: Upper bound
        size: Number of samples

    Returns:
        Array of truncated normal samples
    """
    a = (low - loc) / scale
    b = (high - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=rng)


def clip_biomarker(
        values: np.ndarray,
        marker_type: str,
        ranges: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """
    Clip biomarker values to physiological ranges.

    Args:
        values: Array of biomarker values
        marker_type: Type of biomarker (e.g., 'hdl', 'bmi')
        ranges: Dictionary of (min, max) tuples

    Returns:
        Clipped values
    """
    if marker_type in ranges:
        low, high = ranges[marker_type]
        return np.clip(values, low, high)
    return values


def generate_genetic_risk(
        rng: np.random.Generator,
        n: int,
        disease: str,
        std_params: Dict[str, float],
) -> np.ndarray:
    """
    Generate genetic risk factor from truncated log-normal.

    Args:
        rng: NumPy random number generator
        n: Number of samples
        disease: Disease key (e.g., 'cvd', 'diabetes')
        std_params: Dictionary mapping disease to std dev

    Returns:
        Array of genetic risk factors in [0.5, 2.0]
    """
    sigma = std_params.get(disease, 0.3)
    values = rng.lognormal(0, sigma, n)
    return np.clip(values, 0.5, 2.0)


def calculate_metabolic_rate(
        weight_kg: Union[float, np.ndarray],
        height_cm: Union[float, np.ndarray],
        age_years: Union[int, np.ndarray],
        sex: Union[str, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate BMR using Mifflin-St Jeor formula.

    Supports both scalar and vectorized inputs.
    """
    # Convert to arrays for vectorized operations
    w = np.asarray(weight_kg)
    h = np.asarray(height_cm)
    a = np.asarray(age_years)
    s = np.asarray(sex)

    # Vectorized: True where male
    is_male = s == "M"

    result = np.where(
        is_male,
        10 * w + 6.25 * h - 5 * a + 5,
        10 * w + 6.25 * h - 5 * a - 161
    )

    # Return scalar if input was scalar
    if np.isscalar(weight_kg):
        return float(result)
    return result


def calculate_tdee(
        bmr: Union[float, np.ndarray],
        met_minutes_week: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate Total Daily Energy Expenditure using WHO activity factors.

    Supports both scalar and vectorized inputs.
    """
    b = np.asarray(bmr)
    m = np.asarray(met_minutes_week)

    # Vectorized activity factor selection
    conditions = [
        m < 450,
        (m >= 450) & (m < 900),
        (m >= 900) & (m < 1500),
        m >= 1500,
    ]
    choices = [1.2, 1.375, 1.55, 1.725]

    activity_factor = np.select(conditions, choices)
    result = b * activity_factor

    if np.isscalar(bmr):
        return float(result)
    return result
