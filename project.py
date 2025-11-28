import pandas as pd
import networkx as nx
import itertools
import json
import os
import numpy as np
from collections import Counter, defaultdict
from community import community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set matplotlib style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['legend.facecolor'] = '#16213e'
plt.rcParams['legend.edgecolor'] = '#e94560'
plt.rcParams['font.family'] = 'DejaVu Sans'


# ============================================================
# FAST TSV LOADING HELPERS
# ============================================================
def load_basics_fast(path):
    print("Loading title.basics.tsv (fast mode)...")
    cols = [
        "tconst",
        "titleType",
        "primaryTitle",
        "startYear",
        "runtimeMinutes",
        "genres",
    ]
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=cols,
        dtype={
            "tconst": "string",
            "titleType": "string",
            "primaryTitle": "string",
            "startYear": "string",
            "runtimeMinutes": "string",
            "genres": "string",
        },
        na_values="\\N",
        low_memory=False,
    )
    # type cleaning
    df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    return df


def load_ratings_fast(path):
    print("Loading title.ratings.tsv (fast mode)...")
    cols = ["tconst", "averageRating", "numVotes"]
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=cols,
        dtype={"tconst": "string", "averageRating": "float32", "numVotes": "int32"},
        na_values="\\N",
        low_memory=False,
    )
    return df


def load_principals_fast(path):
    print("Loading title.principals.tsv (fast mode, filtered)...")
    cols = ["tconst", "nconst", "category", "ordering"]
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=cols,
        dtype={
            "tconst": "string",
            "nconst": "string",
            "category": "string",
            "ordering": "int16",
        },
        na_values="\\N",
        low_memory=False,
    )
    # Only actors and actresses
    df = df[df["category"].isin(["actor", "actress"])]
    # Only top-billed
    df = df[df["ordering"] <= 3]
    return df


# ============================================================
# LOAD DATA
# ============================================================
basics = load_basics_fast("title.basics.tsv")
ratings = load_ratings_fast("title.ratings.tsv")
principals = load_principals_fast("title.principals.tsv")

# merge ratings so we can rank by popularity / impact
basics = basics.merge(ratings, on="tconst", how="left")

# ============================================================
# MOVIE FILTERS
# ============================================================
basics = basics[
    (basics["titleType"] == "movie")
    & (basics["runtimeMinutes"] >= 59)
    & (basics["startYear"] >= 1960)
]

# genre parsing
basics["genres"] = basics["genres"].apply(
    lambda g: [] if pd.isna(g) else str(g).split(",")
)

# ============================================================
# CINEMATIC GENRE WHITELIST
# ============================================================
allowed_genres = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
]

# filter movies to only those containing at least one allowed genre
basics = basics[basics["genres"].apply(lambda gl: any(g in allowed_genres for g in gl))]

# ============================================================
# COUNT SINGLE-GENRE MOVIES TO IDENTIFY STRONG GENRES
# ============================================================
single_genre = basics[basics["genres"].apply(lambda x: len(x) == 1)]
genre_counts = Counter(single_genre["genres"].apply(lambda x: x[0]))

valid_genres = [g for g, c in genre_counts.items() if c >= 50]

print("Valid single-genre categories with ≥50 movies:")
print(valid_genres)

# ============================================================
# FILTER PRINCIPALS TO MOVIES IN BASICS
# ============================================================
principals = principals[principals["tconst"].isin(basics["tconst"])]

# ============================================================
# FILTER ACTORS WHO HAVE AT LEAST 3 CREDITS
# ============================================================
actor_credit_counts = principals["nconst"].value_counts()
valid_actors = actor_credit_counts[actor_credit_counts >= 3].index
principals = principals[principals["nconst"].isin(valid_actors)]


# ============================================================
# MACRO-GENRE MAPPING (COMPLETE + SAFE)
# ============================================================
macro_map = {
    # ACTION CLUSTER
    "Action": "ACTION",
    "Adventure": "ACTION",
    "Thriller": "ACTION",
    "Sci-Fi": "ACTION",
    "Western": "ACTION",
    "War": "ACTION",
    # DRAMA CLUSTER
    "Drama": "DRAMA",
    "Romance": "DRAMA",
    "Biography": "DRAMA",
    "History": "DRAMA",
    # COMEDY CLUSTER
    "Comedy": "COMEDY",
    "Music": "COMEDY",
    "Musical": "COMEDY",
    # DARK/CRIME/HORROR CLUSTER
    "Crime": "DARK",
    "Mystery": "DARK",
    "Horror": "DARK",
    "Adult": "DARK",
    # FAMILY / FANTASY / ANIMATION CLUSTER
    "Family": "FAMILY",
    "Animation": "FAMILY",
    "Fantasy": "FAMILY",
    "Sport": "FAMILY",
    "Documentary": "FAMILY",
}


# ============================================================
# HEAVY GENRE EDGE WEIGHTS
# ============================================================
def movie_genre_weight(genre_list):
    n = len(genre_list)
    if n == 1:
        return 1.0
    elif n == 2:
        return 0.25
    elif n == 3:
        return 0.05
    else:
        return 0.01


# ============================================================
# FUNCTION: BUILD GRAPHS FOR GIVEN GENRE SET
# ============================================================
def build_graphs_for_genres(top_genres, sample_size=250):

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

        # assign genre weights to actors
        for actor in actor_list:
            for g in genres:
                actor_genre_weights[actor][g] += w

        # add nodes
        for actor in actor_list:
            if not G_unweighted.has_node(actor):
                G_unweighted.add_node(actor)
            if not G_weighted.has_node(actor):
                G_weighted.add_node(actor)

        # add edges
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
            if g in macro_map:
                actor_macro_weights[actor][macro_map[g]] += v
            else:
                print("WARNING unmapped genre:", g)

    return G_unweighted, G_weighted, actor_genre_weights, actor_macro_weights, movies


# ============================================================
# COMMUNITY DETECTION + MODULARITY + ACCURACY
# ============================================================


def analyze_graphs(
    G_unweighted,
    G_weighted,
    actor_genre_weights,
    actor_macro_weights,
    use_macro=False,
    compute_pagerank=False,
):
    # Select which genre mapping to evaluate
    if use_macro:
        genre_map = actor_macro_weights
    else:
        genre_map = actor_genre_weights

    # ------------------------------------------------------------
    # Louvain on UNWEIGHTED graph
    # ------------------------------------------------------------
    partition_unweighted = community_louvain.best_partition(G_unweighted)
    unweighted_modularity = community_louvain.modularity(
        partition_unweighted, G_unweighted
    )

    # ------------------------------------------------------------
    # Louvain on WEIGHTED graph
    # ------------------------------------------------------------
    partition_weighted = community_louvain.best_partition(G_weighted, weight="weight")
    weighted_modularity = community_louvain.modularity(
        partition_weighted, G_weighted, weight="weight"
    )

    # ------------------------------------------------------------
    # UNWEIGHTED ACCURACY (simple majority vote)
    # ------------------------------------------------------------
    def compute_unweighted_accuracy(partition):
        correct = 0
        total = 0

        # group actors by community
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

        return correct / total if total > 0 else 0

    unweighted_accuracy = compute_unweighted_accuracy(partition_weighted)

    # ------------------------------------------------------------
    # DEGREE WEIGHTED ACCURACY
    # ------------------------------------------------------------
    def compute_degree_weighted_accuracy(partition):
        correct = 0.0
        total = 0.0

        for actor, com in partition.items():
            if actor not in genre_map:
                continue
            if len(genre_map[actor]) == 0:
                continue

            # predicted = community dominant genre
            # compute local distribution:
            com_actors = [a for a, c in partition.items() if c == com]
            counts = Counter()
            for a in com_actors:
                if a in genre_map and len(genre_map[a]) > 0:
                    top = genre_map[a].most_common(1)[0][0]
                    counts[top] += 1

            if len(counts) == 0:
                continue

            predicted = counts.most_common(1)[0][0]

            # degree weight
            deg = G_weighted.degree(actor, weight="weight")
            total += deg
            top_genre = genre_map[actor].most_common(1)[0][0]
            if top_genre == predicted:
                correct += deg

        return correct / total if total > 0 else 0

    degree_accuracy = compute_degree_weighted_accuracy(partition_weighted)

    # ------------------------------------------------------------
    # PAGE RANK WEIGHTED ACCURACY (MACRO ONLY)
    # ------------------------------------------------------------
    pagerank_accuracy = None
    if compute_pagerank:
        pr = nx.pagerank(G_weighted, weight="weight")
        correct = 0.0
        total = 0.0

        # community memberships
        communities = defaultdict(list)
        for actor, com in partition_weighted.items():
            communities[com].append(actor)

        for com, actors in communities.items():

            # find dominant macro genre weighted by pagerank
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

        pagerank_accuracy = correct / total if total > 0 else 0

    return {
        "unweighted_modularity": unweighted_modularity,
        "weighted_modularity": weighted_modularity,
        "unweighted_accuracy": unweighted_accuracy,
        "degree_accuracy": degree_accuracy,
        "pagerank_accuracy": pagerank_accuracy,
        "partition_weighted": partition_weighted,
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_modularity_vs_genres(results_original, results_macro, results_macro_pagerank, output_dir):
    """
    1. Modularity vs. Number of Genres
    One plot for each of the three conditions:
    • Original genres
    • Macro genres
    • Macro genres (PageRank weighted)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_genres = [r["num_genres"] for r in results_original]
    
    # Color palette
    colors = ['#e94560', '#0f3460', '#00d9ff']
    
    # Plot 1: Original Genres
    fig, ax = plt.subplots(figsize=(10, 6))
    unweighted = [r["unweighted_modularity"] for r in results_original]
    weighted = [r["weighted_modularity"] for r in results_original]
    
    ax.plot(num_genres, unweighted, 'o-', color='#e94560', linewidth=2.5, markersize=8, label='Unweighted Graph')
    ax.plot(num_genres, weighted, 's-', color='#00d9ff', linewidth=2.5, markersize=8, label='Weighted Graph')
    ax.set_xlabel('Number of Genres', fontsize=12, fontweight='bold')
    ax.set_ylabel('Modularity', fontsize=12, fontweight='bold')
    ax.set_title('Modularity vs. Number of Genres\n(Original Genres)', fontsize=14, fontweight='bold', color='#e94560')
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(num_genres)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'modularity_original_genres.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    # Plot 2: Macro Genres
    fig, ax = plt.subplots(figsize=(10, 6))
    unweighted = [r["unweighted_modularity"] for r in results_macro]
    weighted = [r["weighted_modularity"] for r in results_macro]
    
    ax.plot(num_genres, unweighted, 'o-', color='#e94560', linewidth=2.5, markersize=8, label='Unweighted Graph')
    ax.plot(num_genres, weighted, 's-', color='#00d9ff', linewidth=2.5, markersize=8, label='Weighted Graph')
    ax.set_xlabel('Number of Genres', fontsize=12, fontweight='bold')
    ax.set_ylabel('Modularity', fontsize=12, fontweight='bold')
    ax.set_title('Modularity vs. Number of Genres\n(Macro Genres)', fontsize=14, fontweight='bold', color='#e94560')
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(num_genres)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'modularity_macro_genres.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    # Plot 3: Macro Genres (PageRank weighted)
    fig, ax = plt.subplots(figsize=(10, 6))
    unweighted = [r["unweighted_modularity"] for r in results_macro_pagerank]
    weighted = [r["weighted_modularity"] for r in results_macro_pagerank]
    
    ax.plot(num_genres, unweighted, 'o-', color='#e94560', linewidth=2.5, markersize=8, label='Unweighted Graph')
    ax.plot(num_genres, weighted, 's-', color='#00d9ff', linewidth=2.5, markersize=8, label='Weighted Graph')
    ax.set_xlabel('Number of Genres', fontsize=12, fontweight='bold')
    ax.set_ylabel('Modularity', fontsize=12, fontweight='bold')
    ax.set_title('Modularity vs. Number of Genres\n(Macro Genres - PageRank Weighted)', fontsize=14, fontweight='bold', color='#e94560')
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(num_genres)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'modularity_macro_pagerank.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    print(f"  Saved modularity plots to {output_dir}")


def plot_accuracy_vs_genres(results_macro_pagerank, output_dir):
    """
    2. Accuracy vs. Number of Genres
    Three curves:
    • Unweighted accuracy
    • Degree-weighted accuracy
    • PageRank-weighted accuracy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_genres = [r["num_genres"] for r in results_macro_pagerank]
    unweighted_acc = [r["unweighted_accuracy"] for r in results_macro_pagerank]
    degree_acc = [r["degree_accuracy"] for r in results_macro_pagerank]
    pagerank_acc = [r["pagerank_accuracy"] for r in results_macro_pagerank]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(num_genres, unweighted_acc, 'o-', color='#e94560', linewidth=2.5, markersize=10, label='Unweighted Accuracy')
    ax.plot(num_genres, degree_acc, 's-', color='#00d9ff', linewidth=2.5, markersize=10, label='Degree-Weighted Accuracy')
    ax.plot(num_genres, pagerank_acc, '^-', color='#f39c12', linewidth=2.5, markersize=10, label='PageRank-Weighted Accuracy')
    
    ax.set_xlabel('Number of Genres', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs. Number of Genres\n(All Weighting Methods)', fontsize=14, fontweight='bold', color='#e94560')
    ax.legend(loc='best', fontsize=11)
    ax.set_xticks(num_genres)
    ax.set_ylim(0, 1)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_genres.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    print(f"  Saved accuracy plot to {output_dir}")


def plot_actor_network(G_weighted, partition, actor_macro_weights, output_dir, max_nodes=300):
    """
    3. Actor Network Visualization (Small Example Graph)
    Use a force layout, colored by Louvain community.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get subgraph for visualization (limit nodes for clarity)
    if len(G_weighted.nodes()) > max_nodes:
        # Get nodes with highest degree for better visualization
        degrees = dict(G_weighted.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
        G_sub = G_weighted.subgraph(top_nodes).copy()
        partition_sub = {n: partition[n] for n in top_nodes if n in partition}
    else:
        G_sub = G_weighted.copy()
        partition_sub = partition.copy()
    
    # Create color map for communities
    communities = set(partition_sub.values())
    cmap_tab20 = plt.colormaps.get_cmap('tab20')
    community_colors = cmap_tab20(np.linspace(0, 1, max(len(communities), 1)))
    color_map = {com: community_colors[i] for i, com in enumerate(sorted(communities))}
    
    node_colors = [color_map.get(partition_sub.get(n, 0), '#888888') for n in G_sub.nodes()]
    
    # Node sizes based on degree
    degrees = dict(G_sub.degree(weight='weight'))
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 400 * (degrees.get(n, 0) / max_deg) for n in G_sub.nodes()]
    
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#0d1b2a')
    fig.patch.set_facecolor('#0d1b2a')
    
    # Spring layout with adjusted parameters for better visualization
    pos = nx.spring_layout(G_sub, k=2/np.sqrt(len(G_sub.nodes())), iterations=50, seed=42)
    
    # Draw edges with transparency
    edge_weights = [G_sub[u][v].get('weight', 1) for u, v in G_sub.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_alphas = [0.1 + 0.3 * (w / max_weight) for w in edge_weights]
    
    nx.draw_networkx_edges(G_sub, pos, alpha=0.15, edge_color='#415a77', width=0.5, ax=ax)
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=node_sizes, 
                           alpha=0.9, ax=ax, linewidths=0.5, edgecolors='white')
    
    # Create legend for communities
    legend_patches = []
    for com in sorted(communities)[:10]:  # Limit legend to 10 communities
        patch = mpatches.Patch(color=color_map[com], label=f'Community {com}')
        legend_patches.append(patch)
    
    ax.legend(handles=legend_patches, loc='upper left', fontsize=9, 
              facecolor='#16213e', edgecolor='#e94560', labelcolor='#eaeaea')
    
    ax.set_title('Actor Network Visualization\n(Colored by Louvain Community)', 
                 fontsize=16, fontweight='bold', color='#e94560')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actor_network.png'), dpi=150, facecolor='#0d1b2a', 
                bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    print(f"  Saved actor network visualization to {output_dir}")


def plot_confusion_matrix(partition, actor_macro_weights, output_dir):
    """
    4. Confusion Matrix Heatmap (Macro Genres Only)
    Rows = predicted community
    Columns = actor's top macro genre
    Cell = percentage overlap
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique macro genres and communities
    macro_genres = sorted(set(['ACTION', 'DRAMA', 'COMEDY', 'DARK', 'FAMILY']))
    communities = sorted(set(partition.values()))
    
    # Build confusion matrix
    matrix = np.zeros((len(communities), len(macro_genres)))
    
    for actor, com in partition.items():
        if actor in actor_macro_weights and len(actor_macro_weights[actor]) > 0:
            top_genre = actor_macro_weights[actor].most_common(1)[0][0]
            if top_genre in macro_genres:
                com_idx = communities.index(com)
                genre_idx = macro_genres.index(top_genre)
                matrix[com_idx, genre_idx] += 1
    
    # Normalize each row to percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_pct = (matrix / row_sums) * 100
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(communities) * 0.5)))
    
    # Custom colormap
    cmap = sns.color_palette("rocket", as_cmap=True)
    
    sns.heatmap(matrix_pct, annot=True, fmt='.1f', cmap=cmap,
                xticklabels=macro_genres, yticklabels=[f'Com {c}' for c in communities],
                ax=ax, cbar_kws={'label': 'Percentage (%)'}, 
                linewidths=0.5, linecolor='#1a1a2e')
    
    ax.set_xlabel('Actor\'s Top Macro Genre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Community', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix: Communities vs. Macro Genres\n(Percentage of Actors)', 
                 fontsize=14, fontweight='bold', color='#e94560')
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_macro.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    print(f"  Saved confusion matrix to {output_dir}")


def plot_pagerank_distribution(G_weighted, output_dir):
    """
    5. PageRank Distribution Histogram
    Distribution of actor PageRanks within the macro graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute PageRank
    pr = nx.pagerank(G_weighted, weight='weight')
    pr_values = list(pr.values())
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create histogram with custom styling
    n, bins, patches = ax.hist(pr_values, bins=50, color='#e94560', alpha=0.8, 
                                edgecolor='#1a1a2e', linewidth=0.5)
    
    # Color gradient for bars
    cm = plt.cm.get_cmap('plasma')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col) if max(col) > 0 else 1
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    ax.set_xlabel('PageRank Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Actors', fontsize=12, fontweight='bold')
    ax.set_title('PageRank Distribution of Actors', fontsize=14, fontweight='bold', color='#e94560')
    
    # Add statistics annotation
    mean_pr = np.mean(pr_values)
    median_pr = np.median(pr_values)
    max_pr = max(pr_values)
    
    stats_text = f'Mean: {mean_pr:.6f}\nMedian: {median_pr:.6f}\nMax: {max_pr:.6f}\nActors: {len(pr_values)}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#e94560', alpha=0.9),
            color='#eaeaea')
    
    ax.set_yscale('log')  # Log scale for better visualization of distribution
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pagerank_distribution.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    print(f"  Saved PageRank distribution to {output_dir}")


def plot_genre_cooccurrence_chord(sampled_movies_df, top_genres, output_dir):
    """
    6. Genre Co-Occurrence Chord Diagram
    Shows which genres frequently appear together in the same movie.
    Uses the sampled movies dataframe (not the full dataset).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the sampled movies directly (already filtered by sample size)
    movies = sampled_movies_df
    
    # Build co-occurrence matrix
    genres = sorted(top_genres)
    n_genres = len(genres)
    cooccurrence = np.zeros((n_genres, n_genres))
    
    for _, row in movies.iterrows():
        movie_genres = [g for g in row['genres'] if g in genres]
        for i, g1 in enumerate(movie_genres):
            for g2 in movie_genres:
                if g1 != g2:
                    idx1 = genres.index(g1)
                    idx2 = genres.index(g2)
                    cooccurrence[idx1, idx2] += 1
    
    # Create a chord-like circular diagram
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    
    # Colors for genres
    cmap_set3 = plt.colormaps.get_cmap('Set3')
    colors = cmap_set3(np.linspace(0, 1, n_genres))
    
    # Position genres around the circle
    angles = np.linspace(0, 2 * np.pi, n_genres, endpoint=False)
    
    # Draw genre labels
    for i, (genre, angle) in enumerate(zip(genres, angles)):
        rotation = np.degrees(angle)
        if angle > np.pi/2 and angle < 3*np.pi/2:
            rotation += 180
            ha = 'right'
        else:
            ha = 'left'
        ax.text(angle, 1.15, genre, ha=ha, va='center', fontsize=11, fontweight='bold',
                rotation=rotation, rotation_mode='anchor', color=colors[i])
    
    # Draw arcs for co-occurrences
    max_cooccur = cooccurrence.max() if cooccurrence.max() > 0 else 1
    
    for i in range(n_genres):
        for j in range(i + 1, n_genres):
            if cooccurrence[i, j] > 0:
                # Draw curved line between genres
                weight = cooccurrence[i, j] / max_cooccur
                alpha = 0.2 + 0.6 * weight
                linewidth = 0.5 + 4 * weight
                
                # Create bezier curve
                theta1, theta2 = angles[i], angles[j]
                
                # Simple arc representation
                theta_range = np.linspace(theta1, theta2, 50)
                r_curve = 0.3 + 0.5 * np.sin(np.linspace(0, np.pi, 50))
                
                # Blend colors
                color = (np.array(colors[i][:3]) + np.array(colors[j][:3])) / 2
                
                ax.plot(theta_range, r_curve, color=color, alpha=alpha, 
                       linewidth=linewidth, solid_capstyle='round')
    
    # Draw genre nodes
    for i, angle in enumerate(angles):
        ax.scatter(angle, 1.0, s=300, c=[colors[i]], zorder=5, edgecolors='white', linewidth=2)
    
    ax.set_ylim(0, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    ax.set_title('Genre Co-Occurrence Diagram\n(Which genres appear together in movies)', 
                 fontsize=16, fontweight='bold', color='#e94560', y=1.08)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_cooccurrence_chord.png'), dpi=150, 
                facecolor='#0d1b2a', bbox_inches='tight')
    plt.close()
    
    # Also create a cleaner heatmap version
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize for percentage
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cooccur_pct = (cooccurrence / row_sums) * 100
    
    mask = np.eye(n_genres, dtype=bool)  # Mask diagonal
    
    sns.heatmap(cooccur_pct, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=genres, yticklabels=genres, ax=ax,
                mask=mask, cbar_kws={'label': 'Co-occurrence %'},
                linewidths=0.5, linecolor='#1a1a2e')
    
    ax.set_title('Genre Co-Occurrence Matrix\n(Percentage of times genres appear together)', 
                 fontsize=14, fontweight='bold', color='#e94560')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_cooccurrence_matrix.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    print(f"  Saved genre co-occurrence diagrams to {output_dir}")


def compute_gini_coefficient(values):
    """
    Compute the Gini coefficient of a distribution.
    Gini = 0 means perfect equality, Gini = 1 means maximum inequality.
    """
    values = np.array(sorted(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    
    # Gini formula: G = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def plot_accuracy_across_samples(cross_sample_data, output_dir):
    """
    1. Accuracy Using PageRank (Across sample sizes)
    Shows how PageRank-weighted accuracy changes as dataset size increases.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_sizes = sorted(cross_sample_data.keys())
    
    # Get the final accuracy (at max genres) for each sample size
    pagerank_accuracies = []
    for ss in sample_sizes:
        # Use the last result (max number of genres) for the final accuracy
        final_result = cross_sample_data[ss]["macro_pagerank"][-1]
        pagerank_accuracies.append(final_result["pagerank_accuracy"])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(sample_sizes, pagerank_accuracies, 'o-', color='#e94560', 
            linewidth=3, markersize=12, markerfacecolor='#00d9ff', 
            markeredgecolor='white', markeredgewidth=2)
    
    # Add value labels
    for i, (ss, acc) in enumerate(zip(sample_sizes, pagerank_accuracies)):
        ax.annotate(f'{acc:.3f}', (ss, acc), textcoords="offset points", 
                   xytext=(0, 15), ha='center', fontsize=11, color='#eaeaea',
                   fontweight='bold')
    
    ax.set_xlabel('Sample Size (Number of Movies)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PageRank-Weighted Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('PageRank Accuracy vs. Dataset Size\n(How prediction improves with more data)', 
                 fontsize=14, fontweight='bold', color='#e94560')
    
    ax.set_xscale('log')
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(s) for s in sample_sizes])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add explanatory text
    ax.text(0.02, 0.98, 
            'Larger datasets → denser networks → more stable hubs\n→ better community-genre alignment',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#e94560', alpha=0.9),
            color='#eaeaea')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_across_samples.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    print(f"  Saved accuracy across samples to {output_dir}")


def plot_pagerank_distribution_comparison(pagerank_data, output_dir):
    """
    2. PageRank Distribution Comparison
    Three comparable histograms showing PageRank distribution evolution.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_sizes = sorted(pagerank_data.keys())
    n_samples = len(sample_sizes)
    
    # Find global min/max for consistent bins across all
    all_pr_values = []
    for ss in sample_sizes:
        all_pr_values.extend(pagerank_data[ss]["pagerank_values"])
    
    # Use same bins for all histograms
    bins = np.logspace(np.log10(min(all_pr_values)), np.log10(max(all_pr_values)), 40)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 6), sharey=True)
    if n_samples == 1:
        axes = [axes]
    
    colors = ['#e94560', '#00d9ff', '#f39c12']
    
    for i, (ss, ax) in enumerate(zip(sample_sizes, axes)):
        pr_values = pagerank_data[ss]["pagerank_values"]
        gini = pagerank_data[ss]["gini"]
        n_actors = len(pr_values)
        
        ax.hist(pr_values, bins=bins, color=colors[i % len(colors)], 
                alpha=0.8, edgecolor='#1a1a2e', linewidth=0.5)
        
        ax.set_xlabel('PageRank Score', fontsize=11, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Number of Actors (log scale)', fontsize=11, fontweight='bold')
        
        ax.set_title(f'{ss} Movies\n({n_actors} actors)', fontsize=12, fontweight='bold', color=colors[i % len(colors)])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add Gini annotation
        ax.text(0.95, 0.95, f'Gini: {gini:.3f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor=colors[i % len(colors)], alpha=0.9),
                color='#eaeaea', fontweight='bold')
        
        ax.set_facecolor('#16213e')
    
    fig.suptitle('PageRank Distribution Across Dataset Sizes\n(Log-log scale reveals power-law structure)', 
                 fontsize=14, fontweight='bold', color='#e94560', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pagerank_distribution_comparison.png'), 
                dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved PageRank distribution comparison to {output_dir}")


def plot_gini_across_samples(pagerank_data, output_dir):
    """
    3. Gini Coefficient of PageRank Distribution across sample sizes.
    Shows how network inequality increases with dataset size.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_sizes = sorted(pagerank_data.keys())
    gini_values = [pagerank_data[ss]["gini"] for ss in sample_sizes]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Bar chart with gradient colors
    bars = ax.bar(range(len(sample_sizes)), gini_values, color='#e94560', 
                  edgecolor='white', linewidth=2)
    
    # Color gradient based on Gini value
    cmap = plt.colormaps.get_cmap('plasma')
    for i, (bar, gini) in enumerate(zip(bars, gini_values)):
        bar.set_facecolor(cmap(gini))
    
    # Add value labels on bars
    for i, (bar, gini) in enumerate(zip(bars, gini_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{gini:.3f}', ha='center', va='bottom', fontsize=12, 
                fontweight='bold', color='#eaeaea')
    
    ax.set_xticks(range(len(sample_sizes)))
    ax.set_xticklabels([str(s) for s in sample_sizes])
    ax.set_xlabel('Sample Size (Number of Movies)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Gini Coefficient of PageRank Distribution\n(Network Inequality vs. Dataset Size)', 
                 fontsize=14, fontweight='bold', color='#e94560')
    
    ax.set_ylim(0, 1)
    ax.axhline(y=0.3, color='#00d9ff', linestyle='--', alpha=0.7, label='Weak network threshold')
    ax.axhline(y=0.6, color='#f39c12', linestyle='--', alpha=0.7, label='Strong hub threshold')
    
    ax.legend(loc='upper left', fontsize=10)
    
    # Add interpretation guide
    ax.text(0.98, 0.5, 
            'Low Gini (< 0.3):\n  Equal importance\n  Weak structure\n\n'
            'High Gini (> 0.6):\n  Strong hubs\n  Power-law network',
            transform=ax.transAxes, fontsize=9, verticalalignment='center',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#e94560', alpha=0.9),
            color='#eaeaea')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gini_across_samples.png'), dpi=150, facecolor='#1a1a2e')
    plt.close()
    
    print(f"  Saved Gini coefficient plot to {output_dir}")


def generate_cross_sample_visualizations(cross_sample_data, pagerank_data, output_dir):
    """Generate visualizations that compare across sample sizes."""
    print(f"\nGenerating cross-sample visualizations in {output_dir}...")
    
    plot_accuracy_across_samples(cross_sample_data, output_dir)
    plot_pagerank_distribution_comparison(pagerank_data, output_dir)
    plot_gini_across_samples(pagerank_data, output_dir)
    
    print(f"  All cross-sample visualizations complete")


def generate_all_visualizations(results_original, results_macro, results_macro_pagerank,
                                 G_weighted, partition, actor_macro_weights,
                                 sampled_movies_df, top_genres, output_dir):
    """Generate all visualizations for a given sample size."""
    print(f"\nGenerating visualizations in {output_dir}...")
    
    plot_modularity_vs_genres(results_original, results_macro, results_macro_pagerank, output_dir)
    plot_accuracy_vs_genres(results_macro_pagerank, output_dir)
    plot_actor_network(G_weighted, partition, actor_macro_weights, output_dir)
    plot_confusion_matrix(partition, actor_macro_weights, output_dir)
    plot_pagerank_distribution(G_weighted, output_dir)
    plot_genre_cooccurrence_chord(sampled_movies_df, top_genres, output_dir)
    
    print(f"  All visualizations complete for {output_dir}")


# ============================================================
# EXPERIMENT LOOP - MULTIPLE SAMPLE SIZES
# ============================================================

# Sample sizes to run experiments for
SAMPLE_SIZES = [250, 1000, 5000]

os.makedirs("./networks", exist_ok=True)
os.makedirs("./diagrams", exist_ok=True)

# Store all results by sample size
all_results = {}

# Store PageRank data for cross-sample visualizations
pagerank_data = {}

for sample_size in SAMPLE_SIZES:
    
    print("\n" + "#" * 80)
    print(f"# RUNNING EXPERIMENTS FOR SAMPLE SIZE: {sample_size} MOVIES")
    print("#" * 80)
    
    results_original = []
    results_macro = []
    results_macro_pagerank = []
    
    # Store the last graph and partition for visualizations (using max genres)
    last_G_weighted = None
    last_partition = None
    last_actor_macro_weights = None
    last_top_genres = None
    
    for num in range(3, 13):
    
        print("\n" + "=" * 80)
        print(f"[Sample={sample_size}] Running experiment with TOP {num} GENRES")
        print("=" * 80)
    
        # Select top N genres by frequency
        top_genres = [g for g, _ in genre_counts.most_common(num)]
        print("Selected genres:", top_genres)
    
        # Build graphs and genre maps
        G_unweighted, G_weighted, actor_genre_weights, actor_macro_weights, sampled_movies = (
            build_graphs_for_genres(top_genres, sample_size=sample_size)
        )
    
        # ========================================================
        # ORIGINAL GENRES ANALYSIS
        # ========================================================
        original_stats = analyze_graphs(
            G_unweighted,
            G_weighted,
            actor_genre_weights,
            actor_macro_weights,
            use_macro=False,
            compute_pagerank=False,
        )
    
        results_original.append(
            {
                "num_genres": num,
                "unweighted_modularity": original_stats["unweighted_modularity"],
                "weighted_modularity": original_stats["weighted_modularity"],
                "unweighted_accuracy": original_stats["unweighted_accuracy"],
                "degree_accuracy": original_stats["degree_accuracy"],
            }
        )
    
        # ========================================================
        # MACRO GENRES ANALYSIS (NO PAGERANK)
        # ========================================================
        macro_stats = analyze_graphs(
            G_unweighted,
            G_weighted,
            actor_genre_weights,
            actor_macro_weights,
            use_macro=True,
            compute_pagerank=False,
        )
    
        results_macro.append(
            {
                "num_genres": num,
                "unweighted_modularity": macro_stats["unweighted_modularity"],
                "weighted_modularity": macro_stats["weighted_modularity"],
                "unweighted_accuracy": macro_stats["unweighted_accuracy"],
                "degree_accuracy": macro_stats["degree_accuracy"],
            }
        )
    
        # ========================================================
        # MACRO GENRES ANALYSIS (PAGERANK)
        # ========================================================
        macro_pagerank_stats = analyze_graphs(
            G_unweighted,
            G_weighted,
            actor_genre_weights,
            actor_macro_weights,
            use_macro=True,
            compute_pagerank=True,
        )
    
        results_macro_pagerank.append(
            {
                "num_genres": num,
                "unweighted_modularity": macro_pagerank_stats["unweighted_modularity"],
                "weighted_modularity": macro_pagerank_stats["weighted_modularity"],
                "unweighted_accuracy": macro_pagerank_stats["unweighted_accuracy"],
                "degree_accuracy": macro_pagerank_stats["degree_accuracy"],
                "pagerank_accuracy": macro_pagerank_stats["pagerank_accuracy"],
            }
        )
    
        # ========================================================
        # SAVE NETWORK FILES
        # ========================================================
        base = f"./networks/s{sample_size}_n_{num}"
    
        # Save GraphML versions (convert lists to strings if needed)
        for G in (G_unweighted, G_weighted):
            for n, d in G.nodes(data=True):
                for k, v in d.items():
                    if isinstance(v, list):
                        G.nodes[n][k] = ",".join(v)
    
        nx.write_graphml(G_unweighted, base + "_unweighted.graphml")
        nx.write_graphml(G_weighted, base + "_weighted.graphml")
    
        # Save JSON metadata (make a copy without partition for JSON serialization)
        metadata = {
            "sample_size": sample_size,
            "num_genres": num,
            "genres": top_genres,
            "original": {k: v for k, v in original_stats.items() if k != "partition_weighted"},
            "macro": {k: v for k, v in macro_stats.items() if k != "partition_weighted"},
            "macro_pagerank": {k: v for k, v in macro_pagerank_stats.items() if k != "partition_weighted"},
        }
    
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    
        print(f"Saved: {base}_unweighted.graphml, {base}_weighted.graphml, {base}.json")
        
        # Store last results for visualizations
        last_G_weighted = G_weighted
        last_partition = macro_pagerank_stats["partition_weighted"]
        last_actor_macro_weights = actor_macro_weights
        last_top_genres = top_genres
        last_sampled_movies = sampled_movies
    
    # Store results for this sample size
    all_results[sample_size] = {
        "original": results_original,
        "macro": results_macro,
        "macro_pagerank": results_macro_pagerank,
    }
    
    # ========================================================
    # GENERATE VISUALIZATIONS FOR THIS SAMPLE SIZE
    # ========================================================
    diagram_dir = f"./diagrams/{sample_size}"
    
    generate_all_visualizations(
        results_original=results_original,
        results_macro=results_macro,
        results_macro_pagerank=results_macro_pagerank,
        G_weighted=last_G_weighted,
        partition=last_partition,
        actor_macro_weights=last_actor_macro_weights,
        sampled_movies_df=last_sampled_movies,
        top_genres=last_top_genres,
        output_dir=diagram_dir
    )
    
    # ========================================================
    # COMPUTE AND STORE PAGERANK DATA FOR CROSS-SAMPLE ANALYSIS
    # ========================================================
    pr = nx.pagerank(last_G_weighted, weight='weight')
    pr_values = list(pr.values())
    gini = compute_gini_coefficient(pr_values)
    
    pagerank_data[sample_size] = {
        "pagerank_values": pr_values,
        "gini": gini,
        "n_actors": len(pr_values),
    }
    
    print(f"\n  PageRank Gini Coefficient for {sample_size} movies: {gini:.4f}")

# ============================================================
# GENERATE CROSS-SAMPLE VISUALIZATIONS
# ============================================================
cross_sample_dir = "./diagrams/across_sample_sizes"
generate_cross_sample_visualizations(all_results, pagerank_data, cross_sample_dir)

# ============================================================
# PRINT FULL SUMMARIES FOR ALL SAMPLE SIZES
# ============================================================

for sample_size in SAMPLE_SIZES:
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUMMARY FOR SAMPLE SIZE: {sample_size}")
    print("=" * 80)
    
    # Print Gini coefficient
    gini = pagerank_data[sample_size]["gini"]
    n_actors = pagerank_data[sample_size]["n_actors"]
    print(f"\nPAGERANK STATISTICS:")
    print(f"  Number of actors: {n_actors}")
    print(f"  Gini Coefficient: {gini:.4f}")
    if gini < 0.3:
        print(f"  Interpretation: Weak network structure (low inequality)")
    elif gini < 0.6:
        print(f"  Interpretation: Moderate hub structure")
    else:
        print(f"  Interpretation: Strong hub structure (power-law network)")
    
    print("\nORIGINAL GENRES (unweighted + degree):")
    for r in all_results[sample_size]["original"]:
        print(r)
    
    print("\nMACRO GENRES (unweighted + degree):")
    for r in all_results[sample_size]["macro"]:
        print(r)
    
    print("\nMACRO GENRES (pagerank weighted):")
    for r in all_results[sample_size]["macro_pagerank"]:
        print(r)

# ============================================================
# SAVE COMBINED RESULTS JSON
# ============================================================
with open("./diagrams/all_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 80)
print("=== ALL EXPERIMENTS COMPLETE ===")
print("=" * 80)
print("\nDiagrams saved to:")
for sample_size in SAMPLE_SIZES:
    print(f"  ./diagrams/{sample_size}/")
print("\nNetwork files saved to: ./networks/")
print("\nCombined results saved to: ./diagrams/all_results.json")
print("\n")
