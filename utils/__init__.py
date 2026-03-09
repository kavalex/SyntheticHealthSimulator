"""
Utility modules for SyntheticHealthSimulator analysis notebooks.
"""
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

__version__ = "1.1.1"
__all__ = [
    'get_version', 'get_data_path', 'get_output_dir',
    'load_baseline', 'load_lifestyle', 'load_biomarkers', 'load_risks', 'load_aggregated', 'load_all_datasets',
    'verify_key_columns',
    'check_range', 'check_correlation', 'check_prevalence',
    'save_figure', 'create_histogram', 'create_scatter', 'create_heatmap', 'create_bar_plot', 'create_line_plot',
    'save_table', 'print_markdown_table',
]
