"""
Visualization functions for network analysis results.

Provides comprehensive plotting functions for modularity, accuracy,
network visualizations, and cross-sample comparisons.
"""

import os
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

from ..config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    COLOR_BACKGROUND,
    FIGURE_FACECOLOR,
    AXES_FACECOLOR,
    TEXT_COLOR,
    LEGEND_FACECOLOR,
    LEGEND_EDGECOLOR,
    MAX_NODES_FOR_VISUALIZATION,
    DPI,
)


def plot_modularity_vs_genres(
    results_original: List[Dict],
    results_macro: List[Dict],
    results_macro_pagerank: List[Dict],
    output_dir: str,
) -> None:
    """
    Plot modularity vs. number of genres for three conditions.

    Creates three separate plots:
    1. Original genres
    2. Macro genres
    3. Macro genres (PageRank weighted)

    Args:
        results_original: List of result dictionaries for original genres
        results_macro: List of result dictionaries for macro genres
        results_macro_pagerank: List of result dictionaries for macro genres with PageRank
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    num_genres = [r["num_genres"] for r in results_original]

    # Plot 1: Original Genres
    fig, ax = plt.subplots(figsize=(10, 6))
    unweighted = [r["unweighted_modularity"] for r in results_original]
    weighted = [r["weighted_modularity"] for r in results_original]

    ax.plot(
        num_genres,
        unweighted,
        "o-",
        color=COLOR_PRIMARY,
        linewidth=2.5,
        markersize=8,
        label="Unweighted Graph",
    )
    ax.plot(
        num_genres,
        weighted,
        "s-",
        color=COLOR_SECONDARY,
        linewidth=2.5,
        markersize=8,
        label="Weighted Graph",
    )
    ax.set_xlabel("Number of Genres", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modularity", fontsize=12, fontweight="bold")
    ax.set_title(
        "Modularity vs. Number of Genres\n(Original Genres)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xticks(num_genres)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "modularity_original_genres.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    # Plot 2: Macro Genres
    fig, ax = plt.subplots(figsize=(10, 6))
    unweighted = [r["unweighted_modularity"] for r in results_macro]
    weighted = [r["weighted_modularity"] for r in results_macro]

    ax.plot(
        num_genres,
        unweighted,
        "o-",
        color=COLOR_PRIMARY,
        linewidth=2.5,
        markersize=8,
        label="Unweighted Graph",
    )
    ax.plot(
        num_genres,
        weighted,
        "s-",
        color=COLOR_SECONDARY,
        linewidth=2.5,
        markersize=8,
        label="Weighted Graph",
    )
    ax.set_xlabel("Number of Genres", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modularity", fontsize=12, fontweight="bold")
    ax.set_title(
        "Modularity vs. Number of Genres\n(Macro Genres)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xticks(num_genres)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "modularity_macro_genres.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    # Plot 3: Macro Genres (PageRank weighted)
    fig, ax = plt.subplots(figsize=(10, 6))
    unweighted = [r["unweighted_modularity"] for r in results_macro_pagerank]
    weighted = [r["weighted_modularity"] for r in results_macro_pagerank]

    ax.plot(
        num_genres,
        unweighted,
        "o-",
        color=COLOR_PRIMARY,
        linewidth=2.5,
        markersize=8,
        label="Unweighted Graph",
    )
    ax.plot(
        num_genres,
        weighted,
        "s-",
        color=COLOR_SECONDARY,
        linewidth=2.5,
        markersize=8,
        label="Weighted Graph",
    )
    ax.set_xlabel("Number of Genres", fontsize=12, fontweight="bold")
    ax.set_ylabel("Modularity", fontsize=12, fontweight="bold")
    ax.set_title(
        "Modularity vs. Number of Genres\n(Macro Genres - PageRank Weighted)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xticks(num_genres)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "modularity_macro_pagerank.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    print(f"  Saved modularity plots to {output_dir}")


def plot_accuracy_vs_genres(
    results_macro_pagerank: List[Dict], output_dir: str
) -> None:
    """
    Plot accuracy vs. number of genres for all weighting methods.

    Args:
        results_macro_pagerank: List of result dictionaries with accuracy metrics
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)

    num_genres = [r["num_genres"] for r in results_macro_pagerank]
    unweighted_acc = [r["unweighted_accuracy"] for r in results_macro_pagerank]
    degree_acc = [r["degree_accuracy"] for r in results_macro_pagerank]
    pagerank_acc = [r["pagerank_accuracy"] for r in results_macro_pagerank]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        num_genres,
        unweighted_acc,
        "o-",
        color=COLOR_PRIMARY,
        linewidth=2.5,
        markersize=10,
        label="Unweighted Accuracy",
    )
    ax.plot(
        num_genres,
        degree_acc,
        "s-",
        color=COLOR_SECONDARY,
        linewidth=2.5,
        markersize=10,
        label="Degree-Weighted Accuracy",
    )
    ax.plot(
        num_genres,
        pagerank_acc,
        "^-",
        color=COLOR_TERTIARY,
        linewidth=2.5,
        markersize=10,
        label="PageRank-Weighted Accuracy",
    )

    ax.set_xlabel("Number of Genres", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Accuracy vs. Number of Genres\n(All Weighting Methods)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )
    ax.legend(loc="best", fontsize=11)
    ax.set_xticks(num_genres)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "accuracy_vs_genres.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    print(f"  Saved accuracy plot to {output_dir}")


def plot_actor_network(
    G_weighted: nx.Graph,
    partition: Dict[str, int],
    actor_macro_weights: Dict,
    output_dir: str,
    max_nodes: int = MAX_NODES_FOR_VISUALIZATION,
) -> None:
    """
    Visualize actor network with community coloring.

    Args:
        G_weighted: Weighted NetworkX graph
        partition: Dictionary mapping actor -> community ID
        actor_macro_weights: Dictionary mapping actor -> genre weights
        output_dir: Directory to save plot
        max_nodes: Maximum number of nodes to visualize
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get subgraph for visualization (limit nodes for clarity)
    if len(G_weighted.nodes()) > max_nodes:
        # Get nodes with highest degree for better visualization
        degrees = dict(G_weighted.degree(weight="weight"))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[
            :max_nodes
        ]
        G_sub = G_weighted.subgraph(top_nodes).copy()
        partition_sub = {n: partition[n] for n in top_nodes if n in partition}
    else:
        G_sub = G_weighted.copy()
        partition_sub = partition.copy()

    # Create color map for communities
    communities = set(partition_sub.values())
    cmap_tab20 = plt.colormaps.get_cmap("tab20")
    community_colors = cmap_tab20(np.linspace(0, 1, max(len(communities), 1)))
    color_map = {com: community_colors[i] for i, com in enumerate(sorted(communities))}

    node_colors = [
        color_map.get(partition_sub.get(n, 0), "#888888") for n in G_sub.nodes()
    ]

    # Node sizes based on degree
    degrees = dict(G_sub.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 400 * (degrees.get(n, 0) / max_deg) for n in G_sub.nodes()]

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor(COLOR_BACKGROUND)
    fig.patch.set_facecolor(COLOR_BACKGROUND)

    # Spring layout with adjusted parameters for better visualization
    pos = nx.spring_layout(
        G_sub, k=2 / np.sqrt(len(G_sub.nodes())), iterations=50, seed=42
    )

    # Draw edges with transparency
    nx.draw_networkx_edges(
        G_sub, pos, alpha=0.15, edge_color="#415a77", width=0.5, ax=ax
    )
    nx.draw_networkx_nodes(
        G_sub,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
        linewidths=0.5,
        edgecolors="white",
    )

    # Create legend for communities
    legend_patches = []
    for com in sorted(communities)[:10]:  # Limit legend to 10 communities
        patch = mpatches.Patch(color=color_map[com], label=f"Community {com}")
        legend_patches.append(patch)

    ax.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=9,
        facecolor=LEGEND_FACECOLOR,
        edgecolor=LEGEND_EDGECOLOR,
        labelcolor=TEXT_COLOR,
    )

    ax.set_title(
        "Actor Network Visualization\n(Colored by Louvain Community)",
        fontsize=16,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "actor_network.png"),
        dpi=DPI,
        facecolor=COLOR_BACKGROUND,
        bbox_inches="tight",
        pad_inches=0.5,
    )
    plt.close()

    print(f"  Saved actor network visualization to {output_dir}")


def plot_confusion_matrix(
    partition: Dict[str, int], actor_macro_weights: Dict, output_dir: str
) -> None:
    """
    Create confusion matrix heatmap showing community vs. macro genre alignment.

    Args:
        partition: Dictionary mapping actor -> community ID
        actor_macro_weights: Dictionary mapping actor -> Counter of macro-genre weights
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique macro genres and communities
    macro_genres = sorted(set(["ACTION", "DRAMA", "COMEDY", "DARK", "FAMILY"]))
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

    sns.heatmap(
        matrix_pct,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        xticklabels=macro_genres,
        yticklabels=[f"Com {c}" for c in communities],
        ax=ax,
        cbar_kws={"label": "Percentage (%)"},
        linewidths=0.5,
        linecolor=FIGURE_FACECOLOR,
    )

    ax.set_xlabel("Actor's Top Macro Genre", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Community", fontsize=12, fontweight="bold")
    ax.set_title(
        "Confusion Matrix: Communities vs. Macro Genres\n(Percentage of Actors)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrix_macro.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    print(f"  Saved confusion matrix to {output_dir}")


def plot_pagerank_distribution(G_weighted: nx.Graph, output_dir: str) -> None:
    """
    Plot histogram of PageRank distribution.

    Args:
        G_weighted: Weighted NetworkX graph
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute PageRank
    pr = nx.pagerank(G_weighted, weight="weight")
    pr_values = list(pr.values())

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create histogram with custom styling
    n, bins, patches = ax.hist(
        pr_values,
        bins=50,
        color=COLOR_PRIMARY,
        alpha=0.8,
        edgecolor=FIGURE_FACECOLOR,
        linewidth=0.5,
    )

    # Color gradient for bars
    cm = plt.cm.get_cmap("plasma")
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col) if max(col) > 0 else 1

    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))

    ax.set_xlabel("PageRank Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Actors", fontsize=12, fontweight="bold")
    ax.set_title(
        "PageRank Distribution of Actors",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )

    # Add statistics annotation
    mean_pr = np.mean(pr_values)
    median_pr = np.median(pr_values)
    max_pr = max(pr_values)

    stats_text = (
        f"Mean: {mean_pr:.6f}\n"
        f"Median: {median_pr:.6f}\n"
        f"Max: {max_pr:.6f}\n"
        f"Actors: {len(pr_values)}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor=AXES_FACECOLOR,
            edgecolor=COLOR_PRIMARY,
            alpha=0.9,
        ),
        color=TEXT_COLOR,
    )

    ax.set_yscale("log")  # Log scale for better visualization of distribution

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "pagerank_distribution.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    print(f"  Saved PageRank distribution to {output_dir}")


def plot_genre_cooccurrence_chord(
    sampled_movies_df: pd.DataFrame, top_genres: List[str], output_dir: str
) -> None:
    """
    Create genre co-occurrence chord diagram and heatmap.

    Args:
        sampled_movies_df: DataFrame of sampled movies with genre information
        top_genres: List of genre names to include
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use the sampled movies directly (already filtered by sample size)
    movies = sampled_movies_df

    # Build co-occurrence matrix
    genres = sorted(top_genres)
    n_genres = len(genres)
    cooccurrence = np.zeros((n_genres, n_genres))

    for _, row in movies.iterrows():
        movie_genres = [g for g in row["genres"] if g in genres]
        for i, g1 in enumerate(movie_genres):
            for g2 in movie_genres:
                if g1 != g2:
                    idx1 = genres.index(g1)
                    idx2 = genres.index(g2)
                    cooccurrence[idx1, idx2] += 1

    # Create a chord-like circular diagram
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)

    # Colors for genres
    cmap_set3 = plt.colormaps.get_cmap("Set3")
    colors = cmap_set3(np.linspace(0, 1, n_genres))

    # Position genres around the circle
    angles = np.linspace(0, 2 * np.pi, n_genres, endpoint=False)

    # Draw genre labels
    for i, (genre, angle) in enumerate(zip(genres, angles)):
        rotation = np.degrees(angle)
        if angle > np.pi / 2 and angle < 3 * np.pi / 2:
            rotation += 180
            ha = "right"
        else:
            ha = "left"
        ax.text(
            angle,
            1.15,
            genre,
            ha=ha,
            va="center",
            fontsize=11,
            fontweight="bold",
            rotation=rotation,
            rotation_mode="anchor",
            color=colors[i],
        )

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

                ax.plot(
                    theta_range,
                    r_curve,
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    solid_capstyle="round",
                )

    # Draw genre nodes
    for i, angle in enumerate(angles):
        ax.scatter(
            angle,
            1.0,
            s=300,
            c=[colors[i]],
            zorder=5,
            edgecolors="white",
            linewidth=2,
        )

    ax.set_ylim(0, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    ax.set_title(
        "Genre Co-Occurrence Diagram\n(Which genres appear together in movies)",
        fontsize=16,
        fontweight="bold",
        color=COLOR_PRIMARY,
        y=1.08,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "genre_cooccurrence_chord.png"),
        dpi=DPI,
        facecolor=COLOR_BACKGROUND,
        bbox_inches="tight",
    )
    plt.close()

    # Also create a cleaner heatmap version
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize for percentage
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cooccur_pct = (cooccurrence / row_sums) * 100

    mask = np.eye(n_genres, dtype=bool)  # Mask diagonal

    sns.heatmap(
        cooccur_pct,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        xticklabels=genres,
        yticklabels=genres,
        ax=ax,
        mask=mask,
        cbar_kws={"label": "Co-occurrence %"},
        linewidths=0.5,
        linecolor=FIGURE_FACECOLOR,
    )

    ax.set_title(
        "Genre Co-Occurrence Matrix\n(Percentage of times genres appear together)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "genre_cooccurrence_matrix.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    print(f"  Saved genre co-occurrence diagrams to {output_dir}")


def plot_accuracy_across_samples(cross_sample_data: Dict, output_dir: str) -> None:
    """
    Plot PageRank accuracy across different sample sizes.

    Args:
        cross_sample_data: Dictionary mapping sample_size -> results
        output_dir: Directory to save plot
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

    ax.plot(
        sample_sizes,
        pagerank_accuracies,
        "o-",
        color=COLOR_PRIMARY,
        linewidth=3,
        markersize=12,
        markerfacecolor=COLOR_SECONDARY,
        markeredgecolor="white",
        markeredgewidth=2,
    )

    # Add value labels
    for i, (ss, acc) in enumerate(zip(sample_sizes, pagerank_accuracies)):
        ax.annotate(
            f"{acc:.3f}",
            (ss, acc),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=11,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    ax.set_xlabel("Sample Size (Number of Movies)", fontsize=12, fontweight="bold")
    ax.set_ylabel("PageRank-Weighted Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "PageRank Accuracy vs. Dataset Size\n(How prediction improves with more data)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )

    ax.set_xscale("log")
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(s) for s in sample_sizes])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add explanatory text
    ax.text(
        0.02,
        0.98,
        "Larger datasets → denser networks → more stable hubs\n"
        "→ better community-genre alignment",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor=AXES_FACECOLOR,
            edgecolor=COLOR_PRIMARY,
            alpha=0.9,
        ),
        color=TEXT_COLOR,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "accuracy_across_samples.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    print(f"  Saved accuracy across samples to {output_dir}")


def plot_pagerank_distribution_comparison(pagerank_data: Dict, output_dir: str) -> None:
    """
    Plot PageRank distribution comparison across sample sizes.

    Args:
        pagerank_data: Dictionary mapping sample_size -> PageRank data
        output_dir: Directory to save plot
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

    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY]

    for i, (ss, ax) in enumerate(zip(sample_sizes, axes)):
        pr_values = pagerank_data[ss]["pagerank_values"]
        gini = pagerank_data[ss]["gini"]
        n_actors = len(pr_values)

        ax.hist(
            pr_values,
            bins=bins,
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor=FIGURE_FACECOLOR,
            linewidth=0.5,
        )

        ax.set_xlabel("PageRank Score", fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_ylabel(
                "Number of Actors (log scale)", fontsize=11, fontweight="bold"
            )

        ax.set_title(
            f"{ss} Movies\n({n_actors} actors)",
            fontsize=12,
            fontweight="bold",
            color=colors[i % len(colors)],
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Add Gini annotation
        ax.text(
            0.95,
            0.95,
            f"Gini: {gini:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor=AXES_FACECOLOR,
                edgecolor=colors[i % len(colors)],
                alpha=0.9,
            ),
            color=TEXT_COLOR,
            fontweight="bold",
        )

        ax.set_facecolor(AXES_FACECOLOR)

    fig.suptitle(
        "PageRank Distribution Across Dataset Sizes\n"
        "(Log-log scale reveals power-law structure)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "pagerank_distribution_comparison.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
        bbox_inches="tight",
    )
    plt.close()

    print(f"  Saved PageRank distribution comparison to {output_dir}")


def plot_gini_across_samples(pagerank_data: Dict, output_dir: str) -> None:
    """
    Plot Gini coefficient across sample sizes.

    Args:
        pagerank_data: Dictionary mapping sample_size -> PageRank data with Gini
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_sizes = sorted(pagerank_data.keys())
    gini_values = [pagerank_data[ss]["gini"] for ss in sample_sizes]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Bar chart with gradient colors
    bars = ax.bar(
        range(len(sample_sizes)),
        gini_values,
        color=COLOR_PRIMARY,
        edgecolor="white",
        linewidth=2,
    )

    # Color gradient based on Gini value
    cmap = plt.colormaps.get_cmap("plasma")
    for i, (bar, gini) in enumerate(zip(bars, gini_values)):
        bar.set_facecolor(cmap(gini))

    # Add value labels on bars
    for i, (bar, gini) in enumerate(zip(bars, gini_values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{gini:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    ax.set_xticks(range(len(sample_sizes)))
    ax.set_xticklabels([str(s) for s in sample_sizes])
    ax.set_xlabel("Sample Size (Number of Movies)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gini Coefficient", fontsize=12, fontweight="bold")
    ax.set_title(
        "Gini Coefficient of PageRank Distribution\n(Network Inequality vs. Dataset Size)",
        fontsize=14,
        fontweight="bold",
        color=COLOR_PRIMARY,
    )

    ax.set_ylim(0, 1)
    ax.axhline(
        y=0.3,
        color=COLOR_SECONDARY,
        linestyle="--",
        alpha=0.7,
        label="Weak network threshold",
    )
    ax.axhline(
        y=0.6,
        color=COLOR_TERTIARY,
        linestyle="--",
        alpha=0.7,
        label="Strong hub threshold",
    )

    ax.legend(loc="upper left", fontsize=10)

    # Add interpretation guide
    ax.text(
        0.98,
        0.5,
        "Low Gini (< 0.3):\n  Equal importance\n  Weak structure\n\n"
        "High Gini (> 0.6):\n  Strong hubs\n  Power-law network",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor=AXES_FACECOLOR,
            edgecolor=COLOR_PRIMARY,
            alpha=0.9,
        ),
        color=TEXT_COLOR,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gini_across_samples.png"),
        dpi=DPI,
        facecolor=FIGURE_FACECOLOR,
    )
    plt.close()

    print(f"  Saved Gini coefficient plot to {output_dir}")


def generate_cross_sample_visualizations(
    cross_sample_data: Dict, pagerank_data: Dict, output_dir: str
) -> None:
    """
    Generate visualizations that compare across sample sizes.

    Args:
        cross_sample_data: Dictionary mapping sample_size -> results
        pagerank_data: Dictionary mapping sample_size -> PageRank data
        output_dir: Directory to save plots
    """
    print(f"\nGenerating cross-sample visualizations in {output_dir}...")

    plot_accuracy_across_samples(cross_sample_data, output_dir)
    plot_pagerank_distribution_comparison(pagerank_data, output_dir)
    plot_gini_across_samples(pagerank_data, output_dir)

    print("  All cross-sample visualizations complete")


def generate_all_visualizations(
    results_original: List[Dict],
    results_macro: List[Dict],
    results_macro_pagerank: List[Dict],
    G_weighted: nx.Graph,
    partition: Dict[str, int],
    actor_macro_weights: Dict,
    sampled_movies_df: pd.DataFrame,
    top_genres: List[str],
    output_dir: str,
) -> None:
    """
    Generate all visualizations for a given sample size.

    Args:
        results_original: Results for original genres
        results_macro: Results for macro genres
        results_macro_pagerank: Results for macro genres with PageRank
        G_weighted: Weighted NetworkX graph
        partition: Community partition
        actor_macro_weights: Actor macro genre weights
        sampled_movies_df: DataFrame of sampled movies
        top_genres: List of top genre names
        output_dir: Directory to save plots
    """
    print(f"\nGenerating visualizations in {output_dir}...")

    plot_modularity_vs_genres(
        results_original, results_macro, results_macro_pagerank, output_dir
    )
    plot_accuracy_vs_genres(results_macro_pagerank, output_dir)
    plot_actor_network(G_weighted, partition, actor_macro_weights, output_dir)
    plot_confusion_matrix(partition, actor_macro_weights, output_dir)
    plot_pagerank_distribution(G_weighted, output_dir)
    plot_genre_cooccurrence_chord(sampled_movies_df, top_genres, output_dir)

    print(f"  All visualizations complete for {output_dir}")
