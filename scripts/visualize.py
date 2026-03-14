#!/usr/bin/env python3
"""
SyntheticHealthSimulator - Biomarker Visualization Script
Generates all 8 plots and validation report from statistics CSV.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.paths import get_output_dir
from src.utils.helpers import filter_biomarker_data
from src.utils.helpers import BIOMARKER_PREFIXES
from src.utils.plots import (
    plot_temporal_trends,
    plot_distribution_evolution,
    plot_percentile_bands,
    plot_delta_distribution,
    plot_dashboard_grid,
    plot_reference_comparison,
    plot_deviation_heatmap,
    generate_summary_report,
)
from src.logging_config import get_logger, setup_logging


def main():
    """Run all visualizations."""
    setup_logging(log_file="output/visualization.log")
    logger = get_logger("synthetic_health")

    output_dir = get_output_dir() / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path to statistics file
    stats_file = get_output_dir() / "statistics" / "all_statistics.csv"
    if not stats_file.exists():
        logger.error(f"Statistics file not found: {stats_file}")
        logger.error("Please run statistics.py first.")
        sys.exit(1)

    logger.info(f"Loading statistics from {stats_file}")
    df_all = pd.read_csv(stats_file)

    # Filter biomarker columns
    df_biomarkers = filter_biomarker_data(df_all, BIOMARKER_PREFIXES)
    if df_biomarkers.empty:
        logger.error("No biomarker data found in statistics file.")
        sys.exit(1)

    logger.info(f"Filtered to {len(df_biomarkers)} biomarker rows")

    # Generate all plots
    logger.info("Generating temporal trends plot...")
    plot_temporal_trends(df_biomarkers, output_dir)

    logger.info("Generating distribution evolution plot...")
    plot_distribution_evolution(df_biomarkers, output_dir)

    logger.info("Generating percentile bands plot...")
    plot_percentile_bands(df_biomarkers, output_dir)

    logger.info("Generating delta distribution plot...")
    plot_delta_distribution(df_biomarkers, output_dir)

    logger.info("Generating dashboard grid plot...")
    plot_dashboard_grid(df_biomarkers, output_dir)

    logger.info("Generating reference comparison plot...")
    _, comparison_df = plot_reference_comparison(df_biomarkers, output_dir)

    logger.info("Generating deviation heatmap...")
    plot_deviation_heatmap(df_biomarkers, output_dir)

    # Generate summary report
    logger.info("Generating summary report...")
    generate_summary_report(df_biomarkers, comparison_df, output_dir)

    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
