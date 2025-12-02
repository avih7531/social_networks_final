"""
Configuration settings for the actor genre network analysis.

Centralizes all configuration values including file paths, sample sizes,
genre mappings, filtering parameters, and visualization settings.
"""

from typing import Dict, List

# ============================================================
# FILE PATHS
# ============================================================
BASICS_PATH = "title.basics.tsv"
RATINGS_PATH = "title.ratings.tsv"
PRINCIPALS_PATH = "title.principals.tsv"

# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================
SAMPLE_SIZES: List[int] = [250, 1000, 5000]
MIN_GENRES: int = 3
MAX_GENRES: int = 12
MIN_GENRE_COUNT: int = 50  # Minimum movies for a genre to be considered valid

# ============================================================
# MOVIE FILTERING PARAMETERS
# ============================================================
MIN_RUNTIME_MINUTES: int = 59
MIN_START_YEAR: int = 1960
MIN_ACTOR_CREDITS: int = 3
TOP_BILLED_ONLY: bool = True
TOP_BILLED_THRESHOLD: int = 3  # Only actors with ordering <= this value

# ============================================================
# GENRE CONFIGURATION
# ============================================================
ALLOWED_GENRES: List[str] = [
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

# Macro-genre mapping for grouping similar genres
MACRO_GENRE_MAP: Dict[str, str] = {
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

# Genre weight function parameters
GENRE_WEIGHTS: Dict[int, float] = {
    1: 1.0,
    2: 0.25,
    3: 0.05,
}

DEFAULT_GENRE_WEIGHT: float = 0.01

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================
OUTPUT_NETWORKS_DIR: str = "./networks"
OUTPUT_DIAGRAMS_DIR: str = "./diagrams"
OUTPUT_RESULTS_FILE: str = "./diagrams/all_results.json"

# ============================================================
# VISUALIZATION SETTINGS
# ============================================================
# Matplotlib styling
PLOT_STYLE: str = "seaborn-v0_8-darkgrid"
FIGURE_FACECOLOR: str = "#1a1a2e"
AXES_FACECOLOR: str = "#16213e"
AXES_EDGECOLOR: str = "#e94560"
TEXT_COLOR: str = "#eaeaea"
GRID_COLOR: str = "#0f3460"
LEGEND_FACECOLOR: str = "#16213e"
LEGEND_EDGECOLOR: str = "#e94560"
FONT_FAMILY: str = "DejaVu Sans"

# Plot colors
COLOR_PRIMARY: str = "#e94560"
COLOR_SECONDARY: str = "#00d9ff"
COLOR_TERTIARY: str = "#f39c12"
COLOR_BACKGROUND: str = "#0d1b2a"

# Network visualization
MAX_NODES_FOR_VISUALIZATION: int = 300
DPI: int = 150

# ============================================================
# DATA LOADING SETTINGS
# ============================================================
# Column selections for efficient loading
BASICS_COLUMNS: List[str] = [
    "tconst",
    "titleType",
    "primaryTitle",
    "startYear",
    "runtimeMinutes",
    "genres",
]

RATINGS_COLUMNS: List[str] = ["tconst", "averageRating", "numVotes"]

PRINCIPALS_COLUMNS: List[str] = ["tconst", "nconst", "category", "ordering"]

# Data types for efficient memory usage
BASICS_DTYPES: Dict[str, str] = {
    "tconst": "string",
    "titleType": "string",
    "primaryTitle": "string",
    "startYear": "string",
    "runtimeMinutes": "string",
    "genres": "string",
}

RATINGS_DTYPES: Dict[str, str] = {
    "tconst": "string",
    "averageRating": "float32",
    "numVotes": "int32",
}

PRINCIPALS_DTYPES: Dict[str, str] = {
    "tconst": "string",
    "nconst": "string",
    "category": "string",
    "ordering": "int16",
}

# ============================================================
# CACHING SETTINGS
# ============================================================
ENABLE_CACHING: bool = True
CACHE_DIR: str = "./.cache"
