"""
Matplotlib styling configuration for consistent visualizations.
"""

import matplotlib.pyplot as plt
import warnings

from ..config import (
    PLOT_STYLE,
    FIGURE_FACECOLOR,
    AXES_FACECOLOR,
    AXES_EDGECOLOR,
    TEXT_COLOR,
    GRID_COLOR,
    LEGEND_FACECOLOR,
    LEGEND_EDGECOLOR,
    FONT_FAMILY,
)


def configure_plot_style() -> None:
    """
    Configure matplotlib with custom dark theme styling.
    
    Sets global matplotlib parameters for consistent, beautiful plots
    with a dark color scheme optimized for presentations.
    """
    warnings.filterwarnings("ignore")
    
    plt.style.use(PLOT_STYLE)
    plt.rcParams["figure.facecolor"] = FIGURE_FACECOLOR
    plt.rcParams["axes.facecolor"] = AXES_FACECOLOR
    plt.rcParams["axes.edgecolor"] = AXES_EDGECOLOR
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    plt.rcParams["grid.color"] = GRID_COLOR
    plt.rcParams["legend.facecolor"] = LEGEND_FACECOLOR
    plt.rcParams["legend.edgecolor"] = LEGEND_EDGECOLOR
    plt.rcParams["font.family"] = FONT_FAMILY

