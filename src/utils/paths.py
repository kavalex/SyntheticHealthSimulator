"""
Path management and configuration for SyntheticHealthSimulator analysis notebooks.
Provides utility functions to locate data directories, retrieve the current version,
and obtain standard output paths for analysis artifacts (figures, tables).
Contains shared settings for plotting and table export.
"""

from pathlib import Path
from typing import Optional, Dict, Any


# Constants
DEFAULT_VERSION_FILE = "VERSION"
DEFAULT_DATA_BASE_DIR = "data"
OUTPUT_DIR_NAME = "output"


# Path resolution helpers

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.
    Assumes that this file is located at src/utils/paths.py relative to the project root.
    """
    # Go up three levels: from src/utils/paths.py -> src/ -> project root
    return Path(__file__).resolve().parent.parent.parent


def get_version(version_file: Optional[str] = None) -> str:
    """
    Read the current version from the VERSION file.

    Args:
        version_file: Path to the VERSION file. If None, defaults to 'VERSION'
                      in the project root.

    Returns:
        Version string prefixed with 'v' (e.g., 'v2.0.0').

    Raises:
        FileNotFoundError: If the version file does not exist.
        ValueError: If the version file is empty.
    """
    if version_file is None:
        version_file = str(get_project_root() / DEFAULT_VERSION_FILE)

    path = Path(version_file)
    if not path.exists():
        raise FileNotFoundError(f"Version file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Version file {path} is empty")

    return f"v{content}"


def get_data_path(version: Optional[str] = None, base_dir: Optional[str] = None) -> Path:
    """
    Return the path to the synthetic data directory for a given version.

    Args:
        version: Version string (e.g., 'v2.0.0'). If None, the current version
                 is read from the VERSION file.
        base_dir: Base directory for data storage. If None, defaults to 'data'
                  under the project root.

    Returns:
        Path object pointing to the versioned data directory.
    """
    if version is None:
        version = get_version()

    if base_dir is None:
        base_dir = str(get_project_root() / DEFAULT_DATA_BASE_DIR)

    return Path(base_dir) / f"synthetic_{version}"


def get_output_dir() -> Path:
    """
    Return the path to the output directory for analysis notebooks.

    The directory is created if it does not exist.

    Returns:
        Path object pointing to the output directory (relative to the current working
        directory, typically the notebook's location). The directory is guaranteed to exist.
    """
    output_dir = Path(OUTPUT_DIR_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


# Shared settings for plotting and tables

PLOT_SETTINGS: Dict[str, Any] = {
    "dpi": 150,
    "figsize_standard": (10, 6),
    "figsize_wide": (14, 8),
    "figsize_large": (18, 10),
    "tight_layout": True,
}

TABLE_SETTINGS: Dict[str, Any] = {
    "index": False,
    "encoding": "utf-8",
    "float_format": "%.4f",
}

def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path,
    dpi: Optional[int] = None,
    tight_layout: bool = True,
    format: str = "svg"
) -> str:
    """
    Save figure with standard settings.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Directory to save to
        dpi: Resolution (default from PLOT_SETTINGS)
        tight_layout: Whether to call tight_layout before saving
        format: File format (svg, png, pdf)

    Returns:
        Full path to saved file
    """
    if dpi is None:
        dpi = PLOT_SETTINGS.get("dpi", 150)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{filename}.{format}"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(filepath, dpi=dpi if format == "png" else None, format=format, bbox_inches='tight')
    plt.close(fig)
    return str(filepath)
import matplotlib.pyplot as plt
