"""
Network analysis functions for community detection and accuracy metrics.

Provides functions for computing modularity, community detection,
and genre prediction accuracy with caching for expensive operations.
"""

from typing import Dict, Optional
from collections import Counter, defaultdict
import networkx as nx
from community import community_louvain

from ..utils.caching import cached_computation


@cached_computation(cache_prefix="pagerank")
def _compute_pagerank_cached(G_weighted: nx.Graph, weight: str = "weight") -> Dict[str, float]:
    """
    Compute PageRank with caching.
    
    Args:
        G_weighted: Weighted NetworkX graph
        weight: Edge weight attribute name
        
    Returns:
        Dictionary mapping node -> PageRank value
    """
    return nx.pagerank(G_weighted, weight=weight)


@cached_computation(cache_prefix="modularity")
def _compute_modularity_cached(
    partition: Dict[str, int],
    G: nx.Graph,
    weight: Optional[str] = None,
) -> float:
    """
    Compute modularity with caching.
    
    Args:
        partition: Dictionary mapping node -> community ID
        G: NetworkX graph
        weight: Edge weight attribute name (None for unweighted)
        
    Returns:
        Modularity score
    """
    if weight:
        return community_louvain.modularity(partition, G, weight=weight)
    else:
        return community_louvain.modularity(partition, G)


def compute_unweighted_accuracy(
    partition: Dict[str, int],
    genre_map: Dict[str, Counter],
) -> float:
    """
    Compute unweighted accuracy using simple majority vote.
    
    For each community, predicts the most common genre among actors.
    Accuracy is the fraction of actors whose top genre matches the
    community's predicted genre.
    
    Args:
        partition: Dictionary mapping actor -> community ID
        genre_map: Dictionary mapping actor -> Counter of genre weights
        
    Returns:
        Accuracy score between 0 and 1
    """
    correct = 0
    total = 0
    
    # Group actors by community
    communities = defaultdict(list)
    for actor, com in partition.items():
        communities[com].append(actor)
    
    for com, actors in communities.items():
        counts = Counter()
        for actor in actors:
            if actor in genre_map:
                # Each actor contributes 1 vote toward their top genre
                if len(genre_map[actor]) > 0:
                    top_genre = genre_map[actor].most_common(1)[0][0]
                    counts[top_genre] += 1
        
        if len(counts) == 0:
            continue
        
        predicted = counts.most_common(1)[0][0]
        
        for actor in actors:
            if actor in genre_map:
                if len(genre_map[actor]) > 0:
                    top_genre = genre_map[actor].most_common(1)[0][0]
                    total += 1
                    if top_genre == predicted:
                        correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_degree_weighted_accuracy(
    partition: Dict[str, int],
    G_weighted: nx.Graph,
    genre_map: Dict[str, Counter],
) -> float:
    """
    Compute degree-weighted accuracy.
    
    Similar to unweighted accuracy, but each actor's vote is weighted
    by their weighted degree in the network.
    
    Args:
        partition: Dictionary mapping actor -> community ID
        G_weighted: Weighted NetworkX graph
        genre_map: Dictionary mapping actor -> Counter of genre weights
        
    Returns:
        Accuracy score between 0 and 1
    """
    correct = 0.0
    total = 0.0
    
    for actor, com in partition.items():
        if actor not in genre_map:
            continue
        if len(genre_map[actor]) == 0:
            continue
        
        # Predicted = community dominant genre
        # Compute local distribution:
        com_actors = [a for a, c in partition.items() if c == com]
        counts = Counter()
        for a in com_actors:
            if a in genre_map and len(genre_map[a]) > 0:
                top = genre_map[a].most_common(1)[0][0]
                counts[top] += 1
        
        if len(counts) == 0:
            continue
        
        predicted = counts.most_common(1)[0][0]
        
        # Degree weight
        deg = G_weighted.degree(actor, weight="weight")
        total += deg
        top_genre = genre_map[actor].most_common(1)[0][0]
        if top_genre == predicted:
            correct += deg
    
    return correct / total if total > 0 else 0.0


def compute_pagerank_weighted_accuracy(
    partition: Dict[str, int],
    G_weighted: nx.Graph,
    genre_map: Dict[str, Counter],
) -> float:
    """
    Compute PageRank-weighted accuracy.
    
    Similar to degree-weighted accuracy, but uses PageRank scores
    instead of degree for weighting actor importance.
    
    Args:
        partition: Dictionary mapping actor -> community ID
        G_weighted: Weighted NetworkX graph
        genre_map: Dictionary mapping actor -> Counter of genre weights
        
    Returns:
        Accuracy score between 0 and 1
    """
    pr = _compute_pagerank_cached(G_weighted, weight="weight")
    correct = 0.0
    total = 0.0
    
    # Community memberships
    communities = defaultdict(list)
    for actor, com in partition.items():
        communities[com].append(actor)
    
    for com, actors in communities.items():
        # Find dominant macro genre weighted by pagerank
        genre_scores = Counter()
        for actor in actors:
            if actor in genre_map and len(genre_map[actor]) > 0:
                top_genre = genre_map[actor].most_common(1)[0][0]
                genre_scores[top_genre] += pr.get(actor, 0)
        
        if len(genre_scores) == 0:
            continue
        
        predicted = genre_scores.most_common(1)[0][0]
        
        for actor in actors:
            if actor in genre_map and len(genre_map[actor]) > 0:
                total += pr.get(actor, 0)
                top_genre = genre_map[actor].most_common(1)[0][0]
                if top_genre == predicted:
                    correct += pr.get(actor, 0)
    
    return correct / total if total > 0 else 0.0


def analyze_graphs(
    G_unweighted: nx.Graph,
    G_weighted: nx.Graph,
    actor_genre_weights: Dict[str, Counter],
    actor_macro_weights: Dict[str, Counter],
    use_macro: bool = False,
    compute_pagerank: bool = False,
) -> Dict:
    """
    Analyze graphs for community structure and genre prediction accuracy.
    
    Performs Louvain community detection on both weighted and unweighted graphs,
    computes modularity scores, and evaluates genre prediction accuracy using
    multiple weighting schemes.
    
    Args:
        G_unweighted: Unweighted NetworkX graph
        G_weighted: Weighted NetworkX graph
        actor_genre_weights: Dictionary mapping actor -> Counter of genre weights
        actor_macro_weights: Dictionary mapping actor -> Counter of macro-genre weights
        use_macro: If True, use macro genres for evaluation; else use original genres
        compute_pagerank: If True, compute PageRank-weighted accuracy (slower)
        
    Returns:
        Dictionary containing:
        - unweighted_modularity: Modularity of unweighted graph
        - weighted_modularity: Modularity of weighted graph
        - unweighted_accuracy: Simple majority vote accuracy
        - degree_accuracy: Degree-weighted accuracy
        - pagerank_accuracy: PageRank-weighted accuracy (None if not computed)
        - partition_weighted: Community partition from weighted graph
    """
    # Select which genre mapping to evaluate
    if use_macro:
        genre_map = actor_macro_weights
    else:
        genre_map = actor_genre_weights
    
    # ------------------------------------------------------------
    # Louvain on UNWEIGHTED graph
    # ------------------------------------------------------------
    partition_unweighted = community_louvain.best_partition(G_unweighted)
    unweighted_modularity = _compute_modularity_cached(
        partition_unweighted, G_unweighted, weight=None
    )
    
    # ------------------------------------------------------------
    # Louvain on WEIGHTED graph
    # ------------------------------------------------------------
    partition_weighted = community_louvain.best_partition(G_weighted, weight="weight")
    weighted_modularity = _compute_modularity_cached(
        partition_weighted, G_weighted, weight="weight"
    )
    
    # ------------------------------------------------------------
    # UNWEIGHTED ACCURACY (simple majority vote)
    # ------------------------------------------------------------
    unweighted_accuracy = compute_unweighted_accuracy(partition_weighted, genre_map)
    
    # ------------------------------------------------------------
    # DEGREE WEIGHTED ACCURACY
    # ------------------------------------------------------------
    degree_accuracy = compute_degree_weighted_accuracy(
        partition_weighted, G_weighted, genre_map
    )
    
    # ------------------------------------------------------------
    # PAGE RANK WEIGHTED ACCURACY (MACRO ONLY)
    # ------------------------------------------------------------
    pagerank_accuracy = None
    if compute_pagerank:
        pagerank_accuracy = compute_pagerank_weighted_accuracy(
            partition_weighted, G_weighted, genre_map
        )
    
    return {
        "unweighted_modularity": unweighted_modularity,
        "weighted_modularity": weighted_modularity,
        "unweighted_accuracy": unweighted_accuracy,
        "degree_accuracy": degree_accuracy,
        "pagerank_accuracy": pagerank_accuracy,
        "partition_weighted": partition_weighted,
    }

