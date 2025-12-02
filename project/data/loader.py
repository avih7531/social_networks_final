"""
Data loading functions for IMDb TSV files.

Provides efficient loading of title.basics, title.ratings, and title.principals
with proper error handling and caching.
"""

from functools import lru_cache
from typing import Tuple
import pandas as pd

from ..config import (
    BASICS_COLUMNS,
    RATINGS_COLUMNS,
    PRINCIPALS_COLUMNS,
    BASICS_DTYPES,
    RATINGS_DTYPES,
    PRINCIPALS_DTYPES,
    BASICS_PATH,
    RATINGS_PATH,
    PRINCIPALS_PATH,
)


@lru_cache(maxsize=1)
def load_basics_fast(path: str = BASICS_PATH) -> pd.DataFrame:
    """
    Load title.basics.tsv file with optimized column selection and dtypes.

    Args:
        path: Path to the title.basics.tsv file

    Returns:
        DataFrame with movie basics information

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        print("Loading title.basics.tsv (fast mode)...")
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=BASICS_COLUMNS,
            dtype=BASICS_DTYPES,
            na_values="\\N",
            low_memory=False,
        )

        # Type cleaning
        df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
        df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")

        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {path}")
    except Exception as e:
        raise ValueError(f"Error loading {path}: {str(e)}")


@lru_cache(maxsize=1)
def load_ratings_fast(path: str = RATINGS_PATH) -> pd.DataFrame:
    """
    Load title.ratings.tsv file with optimized column selection and dtypes.

    Args:
        path: Path to the title.ratings.tsv file

    Returns:
        DataFrame with movie ratings information

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        print("Loading title.ratings.tsv (fast mode)...")
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=RATINGS_COLUMNS,
            dtype=RATINGS_DTYPES,
            na_values="\\N",
            low_memory=False,
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {path}")
    except Exception as e:
        raise ValueError(f"Error loading {path}: {str(e)}")


@lru_cache(maxsize=1)
def load_principals_fast(path: str = PRINCIPALS_PATH) -> pd.DataFrame:
    """
    Load title.principals.tsv file with optimized column selection and dtypes.

    Filters to only actors/actresses with top-billed status (ordering <= 3).

    Args:
        path: Path to the title.principals.tsv file

    Returns:
        DataFrame with principal cast information (actors/actresses only)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    try:
        print("Loading title.principals.tsv (fast mode, filtered)...")
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=PRINCIPALS_COLUMNS,
            dtype=PRINCIPALS_DTYPES,
            na_values="\\N",
            low_memory=False,
        )

        # Only actors and actresses
        df = df[df["category"].isin(["actor", "actress"])]

        # Only top-billed (ordering <= 3)
        df = df[df["ordering"] <= 3]

        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {path}")
    except Exception as e:
        raise ValueError(f"Error loading {path}: {str(e)}")


def load_all_data(
    basics_path: str = BASICS_PATH,
    ratings_path: str = RATINGS_PATH,
    principals_path: str = PRINCIPALS_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required IMDb data files.

    Args:
        basics_path: Path to title.basics.tsv
        ratings_path: Path to title.ratings.tsv
        principals_path: Path to title.principals.tsv

    Returns:
        Tuple of (basics, ratings, principals) DataFrames

    Raises:
        FileNotFoundError: If any required file is missing
        ValueError: If any file format is invalid
    """
    try:
        basics = load_basics_fast(basics_path)
        ratings = load_ratings_fast(ratings_path)
        principals = load_principals_fast(principals_path)

        # Merge ratings into basics
        basics = basics.merge(ratings, on="tconst", how="left")

        return basics, ratings, principals
    except (FileNotFoundError, ValueError):
        raise
