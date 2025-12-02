"""
Main execution script for actor genre network analysis.

Orchestrates data loading, processing, graph building, analysis,
and visualization generation with proper error handling.
"""

import os
import json
import sys
from typing import Dict, List
import networkx as nx

from .config import (
    SAMPLE_SIZES,
    MIN_GENRES,
    MAX_GENRES,
    OUTPUT_NETWORKS_DIR,
    OUTPUT_DIAGRAMS_DIR,
    OUTPUT_RESULTS_FILE,
)
from .data.loader import load_all_data
from .data.processor import process_movie_data, filter_actors, get_top_genres
from .network.builder import build_graphs_for_genres
from .network.analyzer import analyze_graphs
from .visualization.style import configure_plot_style
from .visualization.plots import (
    generate_all_visualizations,
    generate_cross_sample_visualizations,
)
from .utils.metrics import compute_gini_coefficient


def save_network_files(
    G_unweighted: nx.Graph,
    G_weighted: nx.Graph,
    sample_size: int,
    num_genres: int,
    top_genres: List[str],
    original_stats: Dict,
    macro_stats: Dict,
    macro_pagerank_stats: Dict,
) -> None:
    """
    Save network files in GraphML and JSON formats.

    Args:
        G_unweighted: Unweighted NetworkX graph
        G_weighted: Weighted NetworkX graph
        sample_size: Sample size identifier
        num_genres: Number of genres used
        top_genres: List of genre names
        original_stats: Statistics for original genres
        macro_stats: Statistics for macro genres
        macro_pagerank_stats: Statistics for macro genres with PageRank
    """
    base = os.path.join(OUTPUT_NETWORKS_DIR, f"s{sample_size}_n_{num_genres}")

    # Save GraphML versions (convert lists to strings if needed)
    for G in (G_unweighted, G_weighted):
        for n, d in G.nodes(data=True):
            for k, v in d.items():
                if isinstance(v, list):
                    G.nodes[n][k] = ",".join(v)

    try:
        nx.write_graphml(G_unweighted, base + "_unweighted.graphml")
        nx.write_graphml(G_weighted, base + "_weighted.graphml")
    except (IOError, OSError) as e:
        print(f"Warning: Could not save GraphML files: {e}")

    # Save JSON metadata (make a copy without partition for JSON serialization)
    metadata = {
        "sample_size": sample_size,
        "num_genres": num_genres,
        "genres": top_genres,
        "original": {
            k: v for k, v in original_stats.items() if k != "partition_weighted"
        },
        "macro": {k: v for k, v in macro_stats.items() if k != "partition_weighted"},
        "macro_pagerank": {
            k: v for k, v in macro_pagerank_stats.items() if k != "partition_weighted"
        },
    }

    try:
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except (IOError, OSError, ValueError) as e:
        print(f"Warning: Could not save JSON metadata: {e}")

    print(f"Saved: {base}_unweighted.graphml, {base}_weighted.graphml, {base}.json")


def run_experiments() -> None:
    """
    Main function to run all experiments.

    Loads data, processes it, builds graphs for different genre sets,
    performs analysis, and generates visualizations.
    """
    # Configure plot styling
    configure_plot_style()

    # Create output directories
    os.makedirs(OUTPUT_NETWORKS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIAGRAMS_DIR, exist_ok=True)

    # Load data
    try:
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        basics, _, principals = load_all_data()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Process movie data
    try:
        print("\n" + "=" * 80)
        print("PROCESSING DATA")
        print("=" * 80)
        basics, genre_counts = process_movie_data(basics)
    except (ValueError, KeyError, AttributeError) as e:
        print(f"Error processing movie data: {e}")
        sys.exit(1)

    # Filter actors
    try:
        principals = filter_actors(principals, basics["tconst"])
    except (KeyError, AttributeError) as e:  # noqa: BLE001
        print(f"Error filtering actors: {e}")
        sys.exit(1)

    # Store all results by sample size
    all_results: Dict[int, Dict] = {}

    # Store PageRank data for cross-sample visualizations
    pagerank_data: Dict[int, Dict] = {}

    # Run experiments for each sample size
    for sample_size in SAMPLE_SIZES:
        try:
            print("\n" + "#" * 80)
            print(f"# RUNNING EXPERIMENTS FOR SAMPLE SIZE: {sample_size} MOVIES")
            print("#" * 80)

            results_original: List[Dict] = []
            results_macro: List[Dict] = []
            results_macro_pagerank: List[Dict] = []

            # Store the last graph and partition for visualizations (using max genres)
            last_G_weighted = None
            last_partition = None
            last_actor_macro_weights = None
            last_top_genres = None
            last_sampled_movies = None

            for num in range(MIN_GENRES, MAX_GENRES + 1):
                try:
                    print("\n" + "=" * 80)
                    print(
                        f"[Sample={sample_size}] Running experiment with TOP {num} GENRES"
                    )
                    print("=" * 80)

                    # Select top N genres by frequency
                    top_genres = get_top_genres(genre_counts, num)
                    print("Selected genres:", top_genres)

                    # Build graphs and genre maps
                    (
                        G_unweighted,
                        G_weighted,
                        actor_genre_weights,
                        actor_macro_weights,
                        sampled_movies,
                    ) = build_graphs_for_genres(
                        top_genres, basics, principals, sample_size=sample_size
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
                            "unweighted_modularity": original_stats[
                                "unweighted_modularity"
                            ],
                            "weighted_modularity": original_stats[
                                "weighted_modularity"
                            ],
                            "unweighted_accuracy": original_stats[
                                "unweighted_accuracy"
                            ],
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
                            "unweighted_modularity": macro_stats[
                                "unweighted_modularity"
                            ],
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
                            "unweighted_modularity": macro_pagerank_stats[
                                "unweighted_modularity"
                            ],
                            "weighted_modularity": macro_pagerank_stats[
                                "weighted_modularity"
                            ],
                            "unweighted_accuracy": macro_pagerank_stats[
                                "unweighted_accuracy"
                            ],
                            "degree_accuracy": macro_pagerank_stats["degree_accuracy"],
                            "pagerank_accuracy": macro_pagerank_stats[
                                "pagerank_accuracy"
                            ],
                        }
                    )

                    # ========================================================
                    # SAVE NETWORK FILES
                    # ========================================================
                    save_network_files(
                        G_unweighted,
                        G_weighted,
                        sample_size,
                        num,
                        top_genres,
                        original_stats,
                        macro_stats,
                        macro_pagerank_stats,
                    )

                    # Store last results for visualizations
                    last_G_weighted = G_weighted
                    last_partition = macro_pagerank_stats["partition_weighted"]
                    last_actor_macro_weights = actor_macro_weights
                    last_top_genres = top_genres
                    last_sampled_movies = sampled_movies

                except (ValueError, KeyError, AttributeError, RuntimeError) as e:  # noqa: BLE001
                    print(
                        f"Error processing {num} genres for sample size {sample_size}: {e}"
                    )
                    continue

            # Store results for this sample size
            all_results[sample_size] = {
                "original": results_original,
                "macro": results_macro,
                "macro_pagerank": results_macro_pagerank,
            }

            # ========================================================
            # GENERATE VISUALIZATIONS FOR THIS SAMPLE SIZE
            # ========================================================
            if (
                last_G_weighted is not None
                and last_partition is not None
                and last_actor_macro_weights is not None
                and last_top_genres is not None
                and last_sampled_movies is not None
            ):
                diagram_dir = os.path.join(OUTPUT_DIAGRAMS_DIR, str(sample_size))

                try:
                    generate_all_visualizations(
                        results_original=results_original,
                        results_macro=results_macro,
                        results_macro_pagerank=results_macro_pagerank,
                        G_weighted=last_G_weighted,
                        partition=last_partition,
                        actor_macro_weights=last_actor_macro_weights,
                        sampled_movies_df=last_sampled_movies,
                        top_genres=last_top_genres,
                        output_dir=diagram_dir,
                    )
                except (ValueError, KeyError, AttributeError, RuntimeError) as e:  # noqa: BLE001
                    print(f"Error generating visualizations: {e}")

            # ========================================================
            # COMPUTE AND STORE PAGERANK DATA FOR CROSS-SAMPLE ANALYSIS
            # ========================================================
            if last_G_weighted is not None:
                try:
                    pr = nx.pagerank(last_G_weighted, weight="weight")
                    pr_values = list(pr.values())
                    gini = compute_gini_coefficient(pr_values)

                    pagerank_data[sample_size] = {
                        "pagerank_values": pr_values,
                        "gini": gini,
                        "n_actors": len(pr_values),
                    }

                    print(
                        f"\n  PageRank Gini Coefficient for {sample_size} movies: {gini:.4f}"
                    )
                except (ValueError, RuntimeError) as e:  # noqa: BLE001
                    print(f"Error computing PageRank statistics: {e}")

        except (ValueError, KeyError, AttributeError, RuntimeError) as e:  # type: ignore
            print(f"Error processing sample size {sample_size}: {e}")
            continue

    # ============================================================
    # GENERATE CROSS-SAMPLE VISUALIZATIONS
    # ============================================================
    cross_sample_dir = os.path.join(OUTPUT_DIAGRAMS_DIR, "across_sample_sizes")
    try:
        generate_cross_sample_visualizations(
            all_results, pagerank_data, cross_sample_dir
        )
    except (ValueError, KeyError, AttributeError, RuntimeError) as e:  # noqa: PLR1721
        print(f"Error generating cross-sample visualizations: {e}")

    # ============================================================
    # PRINT FULL SUMMARIES FOR ALL SAMPLE SIZES
    # ============================================================
    for sample_size in SAMPLE_SIZES:
        if sample_size not in all_results:
            continue

        print("\n" + "=" * 80)
        print(f"EXPERIMENT SUMMARY FOR SAMPLE SIZE: {sample_size}")
        print("=" * 80)

        # Print Gini coefficient
        if sample_size in pagerank_data:
            gini = pagerank_data[sample_size]["gini"]
            n_actors = pagerank_data[sample_size]["n_actors"]
            print("\nPAGERANK STATISTICS:")  # noqa: F541
            print(f"  Number of actors: {n_actors}")
            print(f"  Gini Coefficient: {gini:.4f}")
            if gini < 0.3:
                print("  Interpretation: Weak network structure (low inequality)")
            elif gini < 0.6:
                print("  Interpretation: Moderate hub structure")
            else:
                print("  Interpretation: Strong hub structure (power-law network)")

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
    try:
        with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
    except (IOError, OSError, ValueError) as e:  # noqa: BLE001
        print(f"Warning: Could not save combined results: {e}")

    print("\n" + "=" * 80)
    print("=== ALL EXPERIMENTS COMPLETE ===")
    print("=" * 80)
    print("\nDiagrams saved to:")
    for sample_size in SAMPLE_SIZES:
        print(f"  {os.path.join(OUTPUT_DIAGRAMS_DIR, str(sample_size))}/")
    print(f"\nNetwork files saved to: {OUTPUT_NETWORKS_DIR}/")
    print(f"\nCombined results saved to: {OUTPUT_RESULTS_FILE}")
    print("\n")


if __name__ == "__main__":
    run_experiments()
