"""
Utility functions for computing network metrics.
"""

from typing import List
import numpy as np


def compute_gini_coefficient(values: List[float]) -> float:
    """
    Compute the Gini coefficient of a distribution.

    The Gini coefficient measures inequality in a distribution:
    - Gini = 0 means perfect equality (all values equal)
    - Gini = 1 means maximum inequality (one value has everything)

    Args:
        values: List of numeric values representing the distribution

    Returns:
        Gini coefficient as a float between 0 and 1

    Examples:
        >>> compute_gini_coefficient([1, 1, 1, 1, 1])
        0.0
        >>> compute_gini_coefficient([0, 0, 0, 0, 100])
        0.8
    """
    if not values:
        return 0.0

    values_array = np.array(sorted(values))
    n = len(values_array)

    if n == 0 or values_array.sum() == 0:
        return 0.0

    # Gini formula: G = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
    index = np.arange(1, n + 1)
    numerator = 2 * np.sum(index * values_array) - (n + 1) * np.sum(values_array)
    denominator = n * np.sum(values_array)

    return numerator / denominator
