"""
Standardized plotting utilities with consistent settings.
"""
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt


def save_figure(
        fig: plt.Figure,
        filename: str,
        output_dir: str = "output",
        dpi: int = 150,
        tight_layout: bool = True,
        verbose: bool = True,
        format: str = "png"
) -> str:
    """
    Save figure with standard settings.

    Args:
        fig: Matplotlib figure
        filename: Output filename (extension added automatically)
        output_dir: Output directory
        dpi: Resolution (only for raster formats like PNG)
        tight_layout: Apply tight layout
        verbose: Print confirmation message
        format: Output format ("png" or "svg")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add extension if not present
    if not filename.endswith(f".{format}"):
        filename = f"{filename}.{format}"

    filepath = output_path / filename

    if tight_layout:
        fig.tight_layout()

    # Save with specified format
    fig.savefig(filepath, dpi=dpi if format == "png" else None, format=format)

    print()

    if verbose:
        file_size = filepath.stat().st_size / 1024  # KB
        print(f"Saved: {filepath} ({file_size:.1f} KB)")

    return str(filepath)


def create_histogram(
        data: List,
        title: str,
        xlabel: str,
        ylabel: str = 'Frequency',
        bins: int = 30,
        figsize: Tuple[int, int] = (10, 6),
        color: str = 'steelblue',
        alpha: float = 0.7,
        **kwargs
) -> plt.Figure:
    """Create and return histogram figure"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, edgecolor='black', alpha=alpha, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


def create_scatter(
        x: List,
        y: List,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = (10, 6),
        alpha: float = 0.5,
        color: str = 'steelblue',
        **kwargs
) -> plt.Figure:
    """Create and return scatter plot figure"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=alpha, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


def create_heatmap(
        data,
        title: str,
        annot: bool = True,
        fmt: str = '.2f',
        cmap: str = 'coolwarm',
        figsize: Tuple[int, int] = (12, 10),
        **kwargs
) -> plt.Figure:
    """Create and return heatmap figure"""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, ax=ax, **kwargs)
    ax.set_title(title)
    return fig


def create_bar_plot(
        categories: List,
        values: List,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = (10, 6),
        color: str = 'steelblue',
        rotation: int = 0,
        **kwargs
) -> plt.Figure:
    """Create and return bar plot figure"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(categories, values, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(categories, rotation=rotation)
    ax.grid(True, alpha=0.3, axis='y')
    return fig


def create_line_plot(
        x: List,
        y: List,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = (10, 6),
        color: str = 'steelblue',
        marker: str = 'o',
        **kwargs
) -> plt.Figure:
    """Create and return line plot figure"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=color, marker=marker, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig
