"""
Data processing and filtering functions.

Provides efficient vectorized operations for filtering movies and actors
based on various criteria.
"""

from typing import List, Tuple
from collections import Counter
import pandas as pd

from ..config import (
    ALLOWED_GENRES,
    MIN_RUNTIME_MINUTES,
    MIN_START_YEAR,
    MIN_ACTOR_CREDITS,
    MIN_GENRE_COUNT,
)


def process_movie_data(
    basics: pd.DataFrame,
) -> Tuple[pd.DataFrame, Counter]:
    """
    Process and filter movie data according to project criteria.
    
    Filters movies by:
    - Type: must be "movie"
    - Runtime: >= MIN_RUNTIME_MINUTES
    - Year: >= MIN_START_YEAR
    - Genres: must contain at least one allowed genre
    
    Also parses genres and identifies valid single-genre categories.
    
    Args:
        basics: DataFrame with movie basics and ratings
        
    Returns:
        Tuple of (filtered_basics, genre_counts) where genre_counts
        is a Counter of single-genre movie counts
    """
    # Filter by movie type, runtime, and year
    basics = basics[
        (basics["titleType"] == "movie")
        & (basics["runtimeMinutes"] >= MIN_RUNTIME_MINUTES)
        & (basics["startYear"] >= MIN_START_YEAR)
    ]
    
    # Parse genres using vectorized operation
    # Convert to string first, then split
    basics["genres"] = (
        basics["genres"]
        .astype(str)
        .str.split(",")
        .apply(lambda x: [g.strip() for g in x if g.strip() != "nan"])
    )
    
    # Filter to movies with at least one allowed genre
    # Use vectorized operation with set intersection
    basics = basics[
        basics["genres"].apply(
            lambda gl: bool(set(gl) & set(ALLOWED_GENRES))
        )
    ]
    
    # Count single-genre movies to identify strong genres
    single_genre = basics[basics["genres"].apply(lambda x: len(x) == 1)]
    genre_counts = Counter(single_genre["genres"].apply(lambda x: x[0]))
    
    # Filter to genres with at least MIN_GENRE_COUNT movies
    valid_genres = [g for g, c in genre_counts.items() if c >= MIN_GENRE_COUNT]
    
    print("Valid single-genre categories with ≥50 movies:")
    print(valid_genres)
    
    return basics, genre_counts


def filter_actors(
    principals: pd.DataFrame,
    valid_movie_ids: pd.Series,
) -> pd.DataFrame:
    """
    Filter principals to only include valid actors in valid movies.
    
    Filters actors by:
    - Must appear in at least MIN_ACTOR_CREDITS movies
    - Must be in movies that passed movie filtering
    
    Args:
        principals: DataFrame with principal cast information
        valid_movie_ids: Series of valid movie tconst values
        
    Returns:
        Filtered DataFrame with only valid actors
    """
    # Filter principals to movies in valid set
    principals = principals[principals["tconst"].isin(valid_movie_ids)]
    
    # Filter actors who have at least MIN_ACTOR_CREDITS
    actor_credit_counts = principals["nconst"].value_counts()
    valid_actors = actor_credit_counts[actor_credit_counts >= MIN_ACTOR_CREDITS].index
    principals = principals[principals["nconst"].isin(valid_actors)]
    
    return principals


def get_top_genres(genre_counts: Counter, num_genres: int) -> List[str]:
    """
    Get the top N genres by frequency.
    
    Args:
        genre_counts: Counter of genre frequencies
        num_genres: Number of top genres to return
        
    Returns:
        List of top genre names
    """
    return [g for g, _ in genre_counts.most_common(num_genres)]

