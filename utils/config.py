"""
Configuration and path management for SyntheticHealthSimulator analysis notebooks.
"""
from pathlib import Path
from typing import Optional


def get_version(version_file: str = "../../VERSION") -> str:
    """Read version from VERSION file"""
    with open(version_file, "r") as f:
        version_raw = f.read().strip()
    return f"v{version_raw}"


def get_data_path(version: Optional[str] = None, base_dir: str = "../../data/") -> Path:
    """Get path to synthetic data directory"""
    if version is None:
        version = get_version()
    return Path(base_dir) / f"synthetic_{version}/"


def get_output_dir() -> Path:
    """Get output directory for current notebook"""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Standard plot settings
PLOT_SETTINGS = {
    "dpi": 150,
    "figsize_standard": (10, 6),
    "figsize_wide": (14, 8),
    "figsize_large": (18, 10),
    "tight_layout": True,
}

# Standard table export settings
TABLE_SETTINGS = {
    "index": False,
    "encoding": "utf-8",
    "float_format": "%.4f",
}
