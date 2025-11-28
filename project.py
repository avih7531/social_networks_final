import pandas as pd
import networkx as nx
import itertools
import json
import os
from collections import Counter, defaultdict
from community import community_louvain
import warnings

warnings.filterwarnings("ignore")


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
    # FAMILY / FANTASY / ANIMATION CLUSTER
    "Family": "FAMILY",
    "Animation": "FAMILY",
    "Fantasy": "FAMILY",
    "Sport": "FAMILY",
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

    return G_unweighted, G_weighted, actor_genre_weights, actor_macro_weights


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
    }


# ============================================================
# EXPERIMENT LOOP
# ============================================================
results_original = []
results_macro = []
results_macro_pagerank = []

os.makedirs("./networks", exist_ok=True)

for num in range(3, 13):

    print("\n" + "=" * 80)
    print(f"Running experiment with TOP {num} GENRES")
    print("=" * 80)

    # Select top N genres by frequency
    top_genres = [g for g, _ in genre_counts.most_common(num)]
    print("Selected genres:", top_genres)

    # Build graphs and genre maps
    G_unweighted, G_weighted, actor_genre_weights, actor_macro_weights = (
        build_graphs_for_genres(top_genres, sample_size=250)
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
    base = f"./networks/n_{num}"

    # Save GraphML versions (convert lists to strings if needed)
    for G in (G_unweighted, G_weighted):
        for n, d in G.nodes(data=True):
            for k, v in d.items():
                if isinstance(v, list):
                    G.nodes[n][k] = ",".join(v)

    nx.write_graphml(G_unweighted, base + "_unweighted.graphml")
    nx.write_graphml(G_weighted, base + "_weighted.graphml")

    # Save JSON metadata
    metadata = {
        "num_genres": num,
        "genres": top_genres,
        "original": original_stats,
        "macro": macro_stats,
        "macro_pagerank": macro_pagerank_stats,
    }

    with open(base + ".json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {base}_unweighted.graphml, {base}_weighted.graphml, {base}.json\n")

# ============================================================
# PRINT FULL SUMMARIES
# ============================================================

print("\n\nEXPERIMENT SUMMARY – ORIGINAL GENRES (unweighted + degree):")
for r in results_original:
    print(r)

print("\n\nEXPERIMENT SUMMARY – MACRO GENRES (unweighted + degree):")
for r in results_macro:
    print(r)

print("\n\nEXPERIMENT SUMMARY – MACRO GENRES (pagerank weighted):")
for r in results_macro_pagerank:
    print(r)

print("\n\n=== END OF EXPERIMENT ===\n")
