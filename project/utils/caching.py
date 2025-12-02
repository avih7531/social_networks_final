"""
Caching utilities for expensive computations.

Provides disk-based caching for graphs and other expensive operations
to avoid recomputation across runs.
"""

import os
import pickle
import hashlib
from typing import Optional, Tuple
from functools import wraps
import networkx as nx

from ..config import ENABLE_CACHING, CACHE_DIR


def _get_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        MD5 hash string as cache key
    """
    key_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


def _ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    if ENABLE_CACHING:
        os.makedirs(CACHE_DIR, exist_ok=True)


def cache_graph(
    sample_size: int,
    genres: Tuple[str, ...],
    graph_type: str,
    graph: nx.Graph,
) -> None:
    """
    Cache a graph to disk.

    Args:
        sample_size: Number of movies in the sample
        genres: Tuple of genre names
        graph_type: Type of graph ('weighted' or 'unweighted')
        graph: NetworkX graph to cache
    """
    if not ENABLE_CACHING:
        return

    _ensure_cache_dir()

    # Create a stable key from parameters
    genre_str = "_".join(sorted(genres))
    cache_key = f"graph_s{sample_size}_{graph_type}_{genre_str}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    try:
        with open(cache_path, "wb") as f:
            pickle.dump(graph, f)
    except Exception:
        # Silently fail if caching doesn't work
        pass


def get_cached_graph(
    sample_size: int,
    genres: Tuple[str, ...],
    graph_type: str,
) -> Optional[nx.Graph]:
    """
    Retrieve a cached graph from disk.

    Args:
        sample_size: Number of movies in the sample
        genres: Tuple of genre names
        graph_type: Type of graph ('weighted' or 'unweighted')

    Returns:
        Cached graph if found, None otherwise
    """
    if not ENABLE_CACHING:
        return None

    _ensure_cache_dir()

    genre_str = "_".join(sorted(genres))
    cache_key = f"graph_s{sample_size}_{graph_type}_{genre_str}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def cached_computation(cache_prefix: str = "comp"):
    """
    Decorator for caching function results to disk.

    Args:
        cache_prefix: Prefix for cache file names

    Returns:
        Decorated function with caching enabled
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_CACHING:
                return func(*args, **kwargs)

            _ensure_cache_dir()

            # Generate cache key
            cache_key = _get_cache_key(*args, **kwargs)
            cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{cache_key}.pkl")

            # Try to load from cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    pass

            # Compute result
            result = func(*args, **kwargs)

            # Save to cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
            except Exception:
                pass

            return result

        return wrapper

    return decorator
