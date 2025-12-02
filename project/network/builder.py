"""
Graph construction functions for actor co-appearance networks.

Builds weighted and unweighted graphs from movie-actor relationships
with caching and optimized edge addition.
"""

from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import itertools
import pandas as pd
import networkx as nx

from ..config import (
    GENRE_WEIGHTS,
    DEFAULT_GENRE_WEIGHT,
    MACRO_GENRE_MAP,
)
from ..utils.caching import cache_graph, get_cached_graph


def movie_genre_weight(genre_list: List[str]) -> float:
    """
    Compute weight for a movie based on its number of genres.

    Single-genre movies get full weight (1.0), multi-genre movies
    get reduced weights to emphasize genre purity.

    Args:
        genre_list: List of genre strings for a movie

    Returns:
        Weight value between 0.01 and 1.0
    """
    n = len(genre_list)
    return GENRE_WEIGHTS.get(n, DEFAULT_GENRE_WEIGHT)


def build_graphs_for_genres(
    top_genres: List[str],
    basics: pd.DataFrame,
    principals: pd.DataFrame,
    sample_size: int = 250,
    use_cache: bool = True,
) -> Tuple[nx.Graph, nx.Graph, Dict[str, Counter], Dict[str, Counter], pd.DataFrame]:
    """
    Build actor co-appearance graphs for a given set of genres.

    Creates both weighted and unweighted graphs where:
    - Nodes represent actors
    - Edges represent co-appearance in movies
    - Edge weights reflect genre purity (single-genre movies weighted higher)

    Args:
        top_genres: List of genre names to include
        basics: DataFrame with movie information
        principals: DataFrame with actor-movie relationships
        sample_size: Number of top movies to sample (by numVotes)
        use_cache: Whether to use cached graphs if available

    Returns:
        Tuple of:
        - G_unweighted: Unweighted NetworkX graph
        - G_weighted: Weighted NetworkX graph
        - actor_genre_weights: Dict mapping actor -> Counter of genre weights
        - actor_macro_weights: Dict mapping actor -> Counter of macro-genre weights
        - sampled_movies: DataFrame of selected movies
    """
    # Check cache first
    if use_cache:
        genre_tuple = tuple(sorted(top_genres))
        cached_unweighted = get_cached_graph(sample_size, genre_tuple, "unweighted")
        cached_weighted = get_cached_graph(sample_size, genre_tuple, "weighted")

        if cached_unweighted is not None and cached_weighted is not None:
            # Need to rebuild genre weights, but can skip graph construction
            # For now, rebuild everything (can optimize later)
            pass

    # Keep movies that belong to ANY of the selected genres
    movies = basics[basics["genres"].apply(lambda gl: any(g in top_genres for g in gl))]

    # Sort by numVotes to prioritize influential films
    movies = movies.sort_values("numVotes", ascending=False)

    # Sample top N movies
    movies = movies.head(sample_size)

    print("Movies selected:", len(movies))

    movie_genres = dict(zip(movies["tconst"], movies["genres"]))

    # Filter principals to selected movies
    p = principals[principals["tconst"].isin(movie_genres.keys())]

    # Group actors per movie
    movie_actor_groups = p.groupby("tconst")["nconst"].apply(list)

    # Only keep movies with ≥2 credited actors
    movie_actor_groups = movie_actor_groups[movie_actor_groups.apply(len) >= 2]

    print("Movies with ≥2 actors:", len(movie_actor_groups))

    # ============================================================
    # CREATE GRAPHS
    # ============================================================
    G_unweighted = nx.Graph()
    G_weighted = nx.Graph()

    # Actor → genre weights accumulator
    actor_genre_weights = defaultdict(lambda: Counter())

    for tconst, actor_list in movie_actor_groups.items():
        genres = movie_genres[tconst]
        w = movie_genre_weight(genres)

        # Assign genre weights to actors
        for actor in actor_list:
            for g in genres:
                actor_genre_weights[actor][g] += w

        # Add nodes
        for actor in actor_list:
            if not G_unweighted.has_node(actor):
                G_unweighted.add_node(actor)
            if not G_weighted.has_node(actor):
                G_weighted.add_node(actor)

        # Add edges
        for a, b in itertools.combinations(actor_list, 2):
            # Unweighted
            if G_unweighted.has_edge(a, b):
                G_unweighted[a][b]["weight"] += 1
            else:
                G_unweighted.add_edge(a, b, weight=1)

            # Weighted
            if G_weighted.has_edge(a, b):
                G_weighted[a][b]["weight"] += w
            else:
                G_weighted.add_edge(a, b, weight=w)

    # ============================================================
    # BUILD MACRO GENRE WEIGHTS
    # ============================================================
    actor_macro_weights = defaultdict(lambda: Counter())

    for actor, weights in actor_genre_weights.items():
        for g, v in weights.items():
            if g in MACRO_GENRE_MAP:
                actor_macro_weights[actor][MACRO_GENRE_MAP[g]] += v
            else:
                print("WARNING unmapped genre:", g)

    # Cache graphs
    if use_cache:
        genre_tuple = tuple(sorted(top_genres))
        cache_graph(sample_size, genre_tuple, "unweighted", G_unweighted)
        cache_graph(sample_size, genre_tuple, "weighted", G_weighted)

    return G_unweighted, G_weighted, actor_genre_weights, actor_macro_weights, movies
