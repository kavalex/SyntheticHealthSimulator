"""
Utility modules for SyntheticHealthSimulator.
"""
# Import updated analysis helpers
from src.utils.analysis_helpers import (
    setup_analysis, save_analysis_table, save_analysis_figure,
    estimate_ou_parameters, detect_stress_events,
    cluster_lifestyle_trajectories,
    estimate_measurement_noise, prepare_prediction_data_pivot,
    calculate_calibration_metrics,
    analyze_missing_pattern, impute_data, run_model_comparison,
    engineer_features, compare_with_reference, run_baseline_models,
    save_descriptive_statistics,
    BIOMARKER_RANGES, EXPECTED_NOISE
)
from src.utils.data_loader import (
    load_baseline, load_lifestyle, load_biomarkers,
    load_risks, load_aggregated, load_all_datasets,
    verify_key_columns
)
from src.utils.helpers import (
    generate_truncnorm, clip_biomarker, generate_genetic_risk,
    calculate_metabolic_rate, calculate_tdee
)
from src.utils.paths import (
    get_version, get_data_path, get_output_dir,
    PLOT_SETTINGS, TABLE_SETTINGS
)
from src.utils.plots import (
    save_figure, create_histogram, create_scatter, create_heatmap,
    create_bar_plot, create_line_plot, create_roc_curve,
    create_precision_recall_curve, create_calibration_curve,
    create_feature_importance_plot
)
from src.utils.tables import save_table, print_markdown_table
# Import consolidated validators
from src.utils.validators import (
    DatasetValidator, ValidationResult, ValidationConfig,
    check_range, check_correlation, check_prevalence,
    run_validation_checks, print_validation_table
)
from .report_generator import build_report, df_to_markdown, generate_mermaid_line_chart, load_all_tables

__all__ = [
    # Paths
    'get_version', 'get_data_path', 'get_output_dir',
    'PLOT_SETTINGS', 'TABLE_SETTINGS',

    # Helpers
    'generate_truncnorm', 'clip_biomarker', 'generate_genetic_risk',
    'calculate_metabolic_rate', 'calculate_tdee',

    # Loaders
    'load_baseline', 'load_lifestyle', 'load_biomarkers',
    'load_risks', 'load_aggregated', 'load_all_datasets',
    'verify_key_columns',

    # Plotting
    'save_figure', 'create_histogram', 'create_scatter', 'create_heatmap',
    'create_bar_plot', 'create_line_plot', 'create_roc_curve',
    'create_precision_recall_curve', 'create_calibration_curve',
    'create_feature_importance_plot',

    # Tables
    'save_table', 'print_markdown_table',

    # Validators
    'DatasetValidator', 'ValidationResult', 'ValidationConfig',
    'check_range', 'check_correlation', 'check_prevalence',
    'run_validation_checks', 'print_validation_table',

    # Analysis Helpers
    'setup_analysis', 'save_analysis_table', 'save_analysis_figure',
    'estimate_ou_parameters', 'detect_stress_events',
    'cluster_lifestyle_trajectories',
    'estimate_measurement_noise', 'prepare_prediction_data_pivot',
    'calculate_calibration_metrics',
    'analyze_missing_pattern', 'impute_data', 'run_model_comparison',
    'engineer_features', 'compare_with_reference', 'run_baseline_models',
    'save_descriptive_statistics',
    'BIOMARKER_RANGES', 'EXPECTED_NOISE',

    # Report Generator
    'build_report', 'df_to_markdown', 'generate_mermaid_line_chart', 'load_all_tables',
]

# Visualization helpers (keeping backward compatibility if needed)
from .plots import (
    plot_temporal_trends,
    plot_distribution_evolution,
    plot_percentile_bands,
    plot_delta_distribution,
    plot_dashboard_grid,
    plot_reference_comparison,
    plot_deviation_heatmap,
    generate_summary_report
)
from .helpers import BIOMARKER_PREFIXES, filter_biomarker_data, get_marker_display_names

__all__ += [
    'plot_temporal_trends',
    'plot_distribution_evolution',
    'plot_percentile_bands',
    'plot_delta_distribution',
    'plot_dashboard_grid',
    'plot_reference_comparison',
    'plot_deviation_heatmap',
    'generate_summary_report',
    'BIOMARKER_PREFIXES',
    'filter_biomarker_data',
    'get_marker_display_names',
]
