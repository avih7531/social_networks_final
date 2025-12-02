# Predicting Movie Genre Using Actor Co-Appearance Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Can we predict a movie's genre using only the structure of actor collaborations—no plot, no text, no semantic features?

This project builds **actor co-appearance networks** from IMDb data and uses **community detection** to test whether genre emerges as a network property. Spoiler: it does. We achieve **61% accuracy** on 5-class genre prediction using only network topology—3x better than random guessing.

## Key Results

| Dataset Size | Actors | PageRank Accuracy | vs. Random |
|:------------:|:------:|:-----------------:|:----------:|
| 250 films | 415 | **76.0%** | 3.8x |
| 1,000 films | 1,174 | **63.9%** | 3.2x |
| 5,000 films | 4,124 | **61.0%** | 3.1x |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/avih7531/social_networks_final/
cd social_networks_final
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download IMDb Data

The project requires three TSV files from IMDb's public datasets:

1. Go to [IMDb Non-Commercial Datasets](https://datasets.imdbws.com/)
2. Download the following files:
   - `title.basics.tsv.gz`
   - `title.ratings.tsv.gz`
   - `title.principals.tsv.gz`

3. **Extract the files** (they come as `.tsv.gz`):
   
   ```bash
   # On Linux/macOS:
   gunzip title.basics.tsv.gz
   gunzip title.ratings.tsv.gz
   gunzip title.principals.tsv.gz
   
   # On Windows, use 7-Zip, WinRAR, or similar
   ```

4. Place the extracted `.tsv` files in the project root directory.

Your directory should look like:
```
social_networks_final/
├── project/              # Modular package structure
├── run_analysis.py       # Main entry point
├── requirements.txt
├── title.basics.tsv      # ~800 MB (you download)
├── title.ratings.tsv     # ~25 MB (you download)
├── title.principals.tsv  # ~2.5 GB (you download)
└── ...
```

### 5. Run the Analysis

```bash
python run_analysis.py
```

**Runtime:** Approximately 3-10 minutes depending on your hardware (the 5,000-film experiments are compute-intensive). Results are cached for faster subsequent runs.

## Project Structure

```
social_networks_final/
│
├── project/                    # Main package
│   ├── __init__.py
│   ├── config.py              # Centralized configuration
│   ├── main.py                # Core orchestration logic
│   │
│   ├── data/                  # Data loading and processing
│   │   ├── __init__.py
│   │   ├── loader.py          # Efficient TSV loading with caching
│   │   └── processor.py       # Data filtering and preprocessing
│   │
│   ├── network/               # Network analysis
│   │   ├── __init__.py
│   │   ├── builder.py         # Graph construction with caching
│   │   └── analyzer.py        # Community detection and metrics
│   │
│   ├── visualization/         # Plotting and visualization
│   │   ├── __init__.py
│   │   ├── style.py           # Matplotlib styling configuration
│   │   └── plots.py           # All visualization functions
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── caching.py          # Disk-based caching utilities
│       └── metrics.py          # Gini coefficient and other metrics
│
├── run_analysis.py            # Entry point script (use this!)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── writeup/                   # Academic writeup
│   ├── writeup.md             # Full paper (Markdown)
│   └── writeup.pdf            # Precompiled PDF version
│
├── presentation/              # LaTeX Beamer slides
│   ├── slides.tex             # Main presentation file
│   ├── slides.pdf             # Compiled presentation (generated)
│   ├── Makefile               # Build instructions
│   └── README.md              # Presentation build guide
│
├── title.basics.tsv           # IMDb movie metadata (you download)
├── title.ratings.tsv          # IMDb ratings data (you download)
├── title.principals.tsv       # IMDb cast/crew data (you download)
│
├── diagrams/                  # Generated visualizations (gitignored)
│   ├── 250/                   # Results for 250-movie sample
│   ├── 1000/                  # Results for 1,000-movie sample
│   ├── 5000/                  # Results for 5,000-movie sample
│   ├── across_sample_sizes/   # Cross-sample comparisons
│   └── all_results.json       # Complete numerical results
│
└── networks/                  # Generated network files (gitignored)
    └── *.graphml              # GraphML files for each configuration
```

## Methodology Overview

1. **Data Filtering**
   - Feature films only (>=59 min runtime)
   - Released 1960 or later
   - Top-3 billed actors/actresses
   - Actors with >=3 film credits

2. **Network Construction**
   - Nodes = actors
   - Edges = co-appearance in a film
   - Edge weights penalize multi-genre films (single-genre = 1.0, dual = 0.25, etc.)
   - Graphs are cached for faster re-runs

3. **Community Detection**
   - Louvain algorithm for modularity optimization
   - Both weighted and unweighted variants
   - Modularity computations are cached

4. **Genre Prediction**
   - 5 macro-genres: ACTION, DRAMA, COMEDY, DARK, FAMILY
   - Three accuracy metrics: unweighted, degree-weighted, PageRank-weighted
   - PageRank computations are cached

5. **Scaling Analysis**
   - Experiments across 250, 1000, and 5000 film samples
   - Gini coefficient to measure network inequality

## Generated Visualizations

| Visualization | Description |
|:--------------|:------------|
| `modularity_*.png` | Modularity vs. number of genres |
| `accuracy_vs_genres.png` | Prediction accuracy across genre counts |
| `actor_network.png` | Force-directed network layout colored by community |
| `confusion_matrix_macro.png` | Community vs. true genre heatmap |
| `pagerank_distribution.png` | Distribution of actor centrality scores |
| `genre_cooccurrence_*.png` | Which genres appear together in films |
| `accuracy_across_samples.png` | How accuracy scales with dataset size |
| `gini_across_samples.png` | Network inequality across scales |

## Customization

### Change Sample Sizes

Edit `SAMPLE_SIZES` in `project/config.py`:

```python
SAMPLE_SIZES: List[int] = [250, 1000, 5000]  # Add or remove sizes
```

### Change Genre Range

Edit `MIN_GENRES` and `MAX_GENRES` in `project/config.py`:

```python
MIN_GENRES: int = 3
MAX_GENRES: int = 12  # Currently tests 3-12 genres
```

### Modify Filtering Parameters

All filtering parameters are in `project/config.py`:

```python
MIN_RUNTIME_MINUTES: int = 59
MIN_START_YEAR: int = 1960
MIN_ACTOR_CREDITS: int = 3
```

### Adjust Visualization Settings

Visualization colors, styles, and output directories are all configurable in `project/config.py`.

## Programmatic Usage

You can also import and use the package programmatically:

```python
from project.main import run_experiments
from project.data import load_all_data, process_movie_data
from project.network import build_graphs_for_genres, analyze_graphs

# Load and process data
basics, _, principals = load_all_data()
basics, genre_counts = process_movie_data(basics)

# Build graphs for specific genres
top_genres = ["Action", "Drama", "Comedy"]
G_unweighted, G_weighted, actor_genre_weights, actor_macro_weights, movies = \
    build_graphs_for_genres(top_genres, basics, principals, sample_size=1000)

# Analyze the graphs
results = analyze_graphs(
    G_unweighted, G_weighted,
    actor_genre_weights, actor_macro_weights,
    use_macro=True,
    compute_pagerank=True
)
```

## Full Writeup

See [`writeup/writeup.md`](writeup/writeup.md) or the precompiled [`writeup/writeup.pdf`](writeup/writeup.pdf) for the complete academic paper including:
- Theoretical background (Louvain, PageRank, Gini coefficient)
- Mathematical formulations
- Detailed results and analysis
- Discussion of findings

## Presentation

A LaTeX Beamer presentation (16:9 aspect ratio) is available in the `presentation/` directory. The slides cover all key findings from the writeup in a presentation-ready format.

### Building the Presentation

**Requirements:**
- LaTeX distribution (TeX Live, MiKTeX, etc.)
- `beamer` and `metropolis` theme packages
- `pdflatex` (or `xelatex` if preferred)

**Quick Build:**
```bash
cd presentation
make
```

This generates `slides.pdf`. See [`presentation/README.md`](presentation/README.md) for detailed build instructions, troubleshooting, and alternative compilation methods.

**Note:** The presentation requires the diagrams to be generated first (run `python run_analysis.py` from the project root).

## Performance Notes

- **Caching**: First run creates cache files in `.cache/` directory. Subsequent runs are significantly faster.
- **Memory Usage:** The principals TSV file is ~2.5 GB. Ensure you have sufficient RAM (8GB+ recommended).
- **Disk Space:** The raw data files require ~3.5 GB. Generated outputs add ~100 MB. Cache files add ~50-100 MB.
- **First Run:** Initial data loading takes 2-5 minutes as pandas parses the large TSV files.
- **Subsequent Runs:** With caching enabled, runs are 40-60% faster.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Genre, fundamentally, is a pattern of relationships.</i>
</p>
