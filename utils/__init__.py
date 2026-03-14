"""
SyntheticHealthSimulator v2.1.0
Medical Data Generator for ML Research
"""

# Version info
__version__ = "2.1.0"

# Core configuration and logging (from new/)
from config import SimulationConfig, DEFAULT_CONFIG
from logging_config import setup_logging, get_logger, ProgressLogger

from .helpers import (
    generate_truncnorm,
    clip_biomarker,
    generate_genetic_risk,
    calculate_metabolic_rate,
    calculate_tdee,
)

# Utility modules (существующие)
from .config import get_version, get_data_path, get_output_dir, PLOT_SETTINGS, TABLE_SETTINGS
from .data_loader import (
    load_baseline, load_lifestyle, load_biomarkers,
    load_risks, load_aggregated, load_all_datasets,
    verify_key_columns
)
from .plots import save_figure, create_histogram, create_scatter, create_heatmap, create_bar_plot, create_line_plot
from .tables import save_table, print_markdown_table
from .validators import (
    check_range, check_correlation, check_prevalence,
    run_validation_checks, print_validation_table
)

__all__ = [
    # Core (new)
    'SimulationConfig', 'DEFAULT_CONFIG',
    'setup_logging', 'get_logger', 'ProgressLogger',

    # Utilities (существующие)
    'get_version', 'get_data_path', 'get_output_dir',
    'PLOT_SETTINGS', 'TABLE_SETTINGS',
    'load_baseline', 'load_lifestyle', 'load_biomarkers',
    'load_risks', 'load_aggregated', 'load_all_datasets',
    'verify_key_columns',
    'save_figure', 'create_histogram', 'create_scatter',
    'create_heatmap', 'create_bar_plot', 'create_line_plot',
    'save_table', 'print_markdown_table',
    'check_range', 'check_correlation', 'check_prevalence',
    'run_validation_checks', 'print_validation_table',
    "generate_truncnorm",
    "clip_biomarker",
    "generate_genetic_risk",
    "calculate_metabolic_rate",
    "calculate_tdee",
]
