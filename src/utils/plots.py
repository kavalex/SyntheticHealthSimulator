import matplotlib.pyplot as plt
"""
Plotting utilities for SyntheticHealthSimulator.
Provides standardized visualization functions for biomarker statistics.
All functions use settings from paths.PLOT_SETTINGS and save via save_figure.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.paths import PLOT_SETTINGS, save_figure
from src.utils.helpers import BIOMARKER_PREFIXES
from src.logging_config import get_logger
from src.coefficients import DEFAULT_COEFFICIENTS

logger = get_logger(__name__)

# Default marker prefixes for biomarker columns

# Years to show in distribution plots
YEARS_TO_SHOW = [0, 5, 10, 15, 19]

# Color palette
COLOR_MAIN = '#2E86AB'
COLOR_REF = 'green'
COLOR_WARN = 'orange'
COLOR_ERROR = 'red'


def _prepare_marker_data(df: pd.DataFrame, marker: str) -> pd.DataFrame:
    """
    Filter and sort data for a single marker.
    """
    marker_data = df[df['marker'] == marker].sort_values('year')
    if marker_data.empty:
        logger.warning(f"No data for marker: {marker}")
    return marker_data


def plot_temporal_trends(
    df: pd.DataFrame,
    output_dir: Path,
    markers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> Optional[plt.Figure]:
    """
    Plot 1: Temporal trends with mean ± std and reference bands.
    """
    if markers is None:
        markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']

    # Filter available markers
    available = [m for m in markers if m in df['marker'].unique()]
    if not available:
        logger.warning("No markers available for temporal trends plot")
        return None

    n_plots = len(available)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, marker in enumerate(available[:4]):  # max 4 subplots
        ax = axes[idx]
        marker_data = _prepare_marker_data(df, marker)
        if marker_data.empty:
            continue

        years = marker_data['year'].values
        mean_vals = marker_data['mean'].values
        std_vals = marker_data['std'].values

        ax.plot(years, mean_vals, 'o-', linewidth=2, label='Synthetic mean', color=COLOR_MAIN)
        ax.fill_between(years, mean_vals - std_vals, mean_vals + std_vals,
                        alpha=0.3, label='±1σ', color=COLOR_MAIN)

        # Reference range if available
        ref = getattr(DEFAULT_COEFFICIENTS.ref, marker, None)
        if ref and 'age_50_59' in ref:
            ref_range = ref['age_50_59']
            ax.axhspan(ref_range['p5'], ref_range['p95'],
                       alpha=0.2, color=COLOR_REF, label='Reference p5-p95 (NHANES)')
            ax.axhline(ref_range['mean'], linestyle='--', color=COLOR_REF,
                       linewidth=2, label=f"Reference mean={ref_range['mean']}")

        # Delta annotation
        if len(marker_data) > 1:
            delta = mean_vals[-1] - mean_vals[0]
            ref_delta = ref.get('delta_20yr', {}).get('mean', 0) if ref else 0
            ax.text(0.02, 0.98, f'Δ = {delta:+.1f} (ref: {ref_delta:+.1f})',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Year')
        ax.set_ylabel(marker)
        ax.set_title(f'{marker} - 20-year trend')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(available), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, '01_temporal_trends', output_dir)
    plt.close(fig)
    return fig


def plot_distribution_evolution(
    df: pd.DataFrame,
    output_dir: Path,
    markers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[plt.Figure]:
    """
    Plot 2: Box-plot style distribution evolution at selected years.
    """
    if markers is None:
        markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']
    if years is None:
        years = YEARS_TO_SHOW

    available = [m for m in markers if m in df['marker'].unique()]
    if not available:
        logger.warning("No markers available for distribution evolution plot")
        return None

    n_plots = len(available)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, marker in enumerate(available[:4]):
        ax = axes[idx]
        marker_data = _prepare_marker_data(df, marker)
        if marker_data.empty:
            continue

        # Prepare data for each year
        plot_data = []
        labels = []
        positions = []

        for i, year in enumerate(years):
            year_row = marker_data[marker_data['year'] == year]
            if year_row.empty:
                continue
            row = year_row.iloc[0]
            # Use percentiles if available, else approximate from mean/std
            p5 = row.get('p5', row['mean'] - 1.96 * row['std'])
            p25 = row.get('p25', row['mean'] - 0.67 * row['std'])
            p50 = row.get('median', row['mean'])
            p75 = row.get('p75', row['mean'] + 0.67 * row['std'])
            p95 = row.get('p95', row['mean'] + 1.96 * row['std'])

            plot_data.append([p5, p25, p50, p75, p95])
            labels.append(f'Year {year}')
            positions.append(i)

        # Draw custom boxplot
        for i, (pos, data) in enumerate(zip(positions, plot_data)):
            p5, p25, p50, p75, p95 = data
            # Whiskers
            ax.plot([pos, pos], [p5, p95], color=COLOR_MAIN, linewidth=1)
            # Box
            ax.plot([pos-0.15, pos+0.15], [p25, p25], color=COLOR_MAIN, linewidth=2)
            ax.plot([pos-0.15, pos+0.15], [p75, p75], color=COLOR_MAIN, linewidth=2)
            ax.plot([pos-0.15, pos-0.15], [p25, p75], color=COLOR_MAIN, linewidth=2)
            ax.plot([pos+0.15, pos+0.15], [p25, p75], color=COLOR_MAIN, linewidth=2)
            # Median
            ax.plot([pos-0.2, pos+0.2], [p50, p50], color='red', linewidth=2)

        # Reference range
        ref = getattr(DEFAULT_COEFFICIENTS.ref, marker, None)
        if ref and 'age_50_59' in ref:
            ax.axhspan(ref['age_50_59']['p5'], ref['age_50_59']['p95'],
                       alpha=0.2, color=COLOR_REF, label='Reference p5-p95')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(marker)
        ax.set_title(f'{marker} - distribution evolution')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    for idx in range(len(available), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, '02_distribution_evolution', output_dir)
    plt.close(fig)
    return fig


def plot_percentile_bands(
    df: pd.DataFrame,
    output_dir: Path,
    markers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[plt.Figure]:
    """
    Plot 3: Percentile bands (p5, p25, p50, p75, p95) over years.
    """
    if markers is None:
        markers = ['sbp_mmhg', 'hba1c_percent']

    available = [m for m in markers if m in df['marker'].unique()]
    if not available:
        logger.warning("No markers available for percentile bands plot")
        return None

    fig, axes = plt.subplots(len(available), 1, figsize=(figsize[0], 6*len(available)))
    if len(available) == 1:
        axes = [axes]

    for idx, marker in enumerate(available):
        ax = axes[idx]
        marker_data = _prepare_marker_data(df, marker)
        if marker_data.empty:
            continue

        years = marker_data['year'].values
        p5 = marker_data['p5'].values if 'p5' in marker_data else marker_data['mean'] - 1.96*marker_data['std']
        p25 = marker_data['p25'].values if 'p25' in marker_data else marker_data['mean'] - 0.67*marker_data['std']
        p50 = marker_data['median'].values if 'median' in marker_data else marker_data['mean']
        p75 = marker_data['p75'].values if 'p75' in marker_data else marker_data['mean'] + 0.67*marker_data['std']
        p95 = marker_data['p95'].values if 'p95' in marker_data else marker_data['mean'] + 1.96*marker_data['std']

        ax.fill_between(years, p5, p95, alpha=0.3, color=COLOR_MAIN, label='p5-p95')
        ax.fill_between(years, p25, p75, alpha=0.5, color=COLOR_MAIN, label='p25-p75')
        ax.plot(years, p50, 'k-', linewidth=2, label='Median')

        ref = getattr(DEFAULT_COEFFICIENTS.ref, marker, None)
        if ref and 'age_50_59' in ref:
            ax.axhspan(ref['age_50_59']['p5'], ref['age_50_59']['p95'],
                       alpha=0.2, color=COLOR_REF, linestyle='--', label='Reference p5-p95')

        ax.set_xlabel('Year')
        ax.set_ylabel(marker)
        ax.set_title(f'{marker} - percentile bands')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, '03_percentile_bands', output_dir)
    plt.close(fig)
    return fig


def plot_delta_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    markers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[plt.Figure]:
    """
    Plot 4: Distribution of changes from year 0 to year 19 (simulated as normal).
    """
    if markers is None:
        markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']

    available = [m for m in markers if m in df['marker'].unique()]
    if not available:
        logger.warning("No markers available for delta distribution plot")
        return None

    n_plots = len(available)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, marker in enumerate(available[:4]):
        ax = axes[idx]
        marker_data = _prepare_marker_data(df, marker)
        if marker_data.empty:
            continue

        year0 = marker_data[marker_data['year'] == 0]
        year19 = marker_data[marker_data['year'] == 19]
        if year0.empty or year19.empty:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            continue

        mean0 = year0['mean'].values[0]
        std0 = year0['std'].values[0]
        mean19 = year19['mean'].values[0]
        std19 = year19['std'].values[0]

        delta_mean = mean19 - mean0
        delta_std = np.sqrt(std0**2 + std19**2)

        # Reference delta
        ref = getattr(DEFAULT_COEFFICIENTS.ref, marker, None)
        if ref:
            ref_delta = ref.get('delta_20yr', {}).get('mean', 0)
            ref_min = ref.get('delta_20yr', {}).get('min', ref_delta - 5)
            ref_max = ref.get('delta_20yr', {}).get('max', ref_delta + 5)
        else:
            ref_delta, ref_min, ref_max = 0, -5, 5

        # Plot normal distribution
        x = np.linspace(delta_mean - 3*delta_std, delta_mean + 3*delta_std, 200)
        y = stats.norm.pdf(x, delta_mean, delta_std)
        ax.fill_between(x, y, alpha=0.5, color=COLOR_MAIN, label='Synthetic Δ distribution')
        ax.plot(x, y, color=COLOR_MAIN, linewidth=2)

        ax.axvline(delta_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Synthetic Δ = {delta_mean:+.1f}')
        ax.axvline(ref_delta, color=COLOR_REF, linestyle='-.', linewidth=2,
                   label=f'Reference Δ = {ref_delta:+.1f}')
        ax.axvspan(ref_min, ref_max, alpha=0.2, color=COLOR_REF, label='Reference range')

        in_range = ref_min <= delta_mean <= ref_max
        status = 'OK' if in_range else 'WARN'
        ax.text(0.02, 0.98, f'Status: {status}', transform=ax.transAxes,
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if in_range else 'lightcoral', alpha=0.5))

        ax.set_xlabel('Change over 20 years')
        ax.set_ylabel('Density')
        ax.set_title(f'{marker} - change distribution')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(available), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, '04_delta_distribution', output_dir)
    plt.close(fig)
    return fig


def plot_dashboard_grid(
    df: pd.DataFrame,
    output_dir: Path,
    markers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> Optional[plt.Figure]:
    """
    Plot 5: Dashboard grid (2x3) of all main biomarkers with trends.
    """
    if markers is None:
        markers = BIOMARKER_PREFIXES

    available = [m for m in markers if m in df['marker'].unique()]
    if not available:
        logger.warning("No markers available for dashboard grid")
        return None

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, marker in enumerate(available[:6]):
        ax = axes[idx]
        marker_data = _prepare_marker_data(df, marker)
        if marker_data.empty:
            continue

        years = marker_data['year'].values
        mean_vals = marker_data['mean'].values
        std_vals = marker_data['std'].values

        ax.plot(years, mean_vals, 'o-', linewidth=2, color=COLOR_MAIN)
        ax.fill_between(years, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, color=COLOR_MAIN)

        # Reference line
        ref = getattr(DEFAULT_COEFFICIENTS.ref, marker, None)
        if ref and 'age_50_59' in ref:
            ax.axhline(ref['age_50_59']['mean'], color=COLOR_REF, linestyle='--', linewidth=1.5, alpha=0.7)

        delta = mean_vals[-1] - mean_vals[0]
        ax.text(0.02, 0.95, f'Δ = {delta:+.1f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Year')
        ax.set_ylabel(marker)
        ax.set_title(marker)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 20)

    # Hide unused subplots
    for idx in range(len(available), 6):
        axes[idx].set_visible(False)

    plt.suptitle('SyntheticHealthSimulator - Biomarker Dashboard (20 years)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, '05_dashboard_grid', output_dir)
    plt.close(fig)
    return fig


def plot_reference_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    markers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> Optional[plt.Figure]:
    """
    Plot 6: Comparison of synthetic year 19 vs reference values.
    Also returns comparison table as DataFrame.
    """
    if markers is None:
        markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']

    available = [m for m in markers if m in df['marker'].unique()]
    if not available:
        logger.warning("No markers available for reference comparison")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    comparison_rows = []

    for idx, marker in enumerate(available[:4]):
        ax = axes[idx]
        marker_data = _prepare_marker_data(df, marker)
        if marker_data.empty:
            continue

        year19 = marker_data[marker_data['year'] == 19]
        if year19.empty:
            continue

        syn_mean = year19['mean'].values[0]
        syn_std = year19['std'].values[0]

        ref = getattr(DEFAULT_COEFFICIENTS.ref, marker, None)
        if not ref or 'age_50_59' not in ref:
            ax.text(0.5, 0.5, 'No reference', ha='center', va='center')
            continue

        ref_mean = ref['age_50_59']['mean']
        ref_p5 = ref['age_50_59']['p5']
        ref_p95 = ref['age_50_59']['p95']

        # Reference range
        ax.axvspan(ref_p5, ref_p95, alpha=0.3, color=COLOR_REF, label=f'Reference p5-p95: {ref_p5}-{ref_p95}')
        ax.axvline(ref_mean, color=COLOR_REF, linestyle='--', linewidth=2, label=f'Reference mean: {ref_mean}')

        # Synthetic
        ax.axvline(syn_mean, color=COLOR_MAIN, linestyle='-', linewidth=3, label=f'Synthetic mean: {syn_mean:.1f}')
        ax.axvspan(syn_mean - syn_std, syn_mean + syn_std, alpha=0.3, color=COLOR_MAIN, label=f'Synthetic ±σ: {syn_std:.1f}')

        # t-test approximation
        t_stat = (syn_mean - ref_mean) / (syn_std / np.sqrt(5000))  # assuming n=5000
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        in_range = ref_p5 <= syn_mean <= ref_p95
        status = 'IN RANGE' if in_range else 'OUT OF RANGE'
        ax.text(0.02, 0.98, f'{status}\np ≈ {p_val:.4f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if in_range else 'lightcoral', alpha=0.5))

        ax.set_xlabel(marker)
        ax.set_ylabel('Density')
        ax.set_title(f'{marker} - Year 19 vs Reference')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        comparison_rows.append({
            'marker': marker,
            'synthetic_mean': syn_mean,
            'synthetic_std': syn_std,
            'reference_mean': ref_mean,
            'reference_p5': ref_p5,
            'reference_p95': ref_p95,
            't_statistic': t_stat,
            'p_value': p_val,
            'in_range': in_range
        })

    for idx in range(len(available), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig, '06_reference_comparison', output_dir)

    # Save comparison table
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(output_dir / 'reference_comparison_table.csv', index=False)
        logger.info(f"Saved comparison table to {output_dir / 'reference_comparison_table.csv'}")
    else:
        comp_df = pd.DataFrame()

    plt.close(fig)
    return fig, comp_df


def plot_deviation_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    markers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[plt.Figure]:
    """
    Plot 7: Heatmap of percentage deviation from reference mean across years.
    """
    if markers is None:
        markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']
    if years is None:
        years = list(range(0, 20, 2))  # every 2 years

    available_markers = [m for m in markers if m in df['marker'].unique()]
    if not available_markers:
        logger.warning("No markers available for deviation heatmap")
        return None

    # Build deviation matrix
    matrix = []
    row_labels = []

    for marker in available_markers:
        ref = getattr(DEFAULT_COEFFICIENTS.ref, marker, None)
        if not ref or 'age_50_59' not in ref:
            continue
        ref_mean = ref['age_50_59']['mean']
        marker_data = _prepare_marker_data(df, marker)
        if marker_data.empty:
            continue

        row = []
        for year in years:
            yr_data = marker_data[marker_data['year'] == year]
            if not yr_data.empty:
                syn_mean = yr_data['mean'].values[0]
                dev_pct = (syn_mean - ref_mean) / ref_mean * 100
                row.append(dev_pct)
            else:
                row.append(np.nan)
        matrix.append(row)
        row_labels.append(marker)

    if not matrix:
        logger.warning("No deviation data for heatmap")
        return None

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f'Y{y}' for y in years])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                color = 'black' if abs(matrix[i, j]) < 10 else 'white'
                ax.text(j, i, f'{matrix[i, j]:+.1f}%', ha='center', va='center', fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Deviation from reference (%)')

    ax.set_title('Biomarker deviation from reference mean (NHANES age 50-59)')
    plt.tight_layout()
    save_figure(fig, '07_deviation_heatmap', output_dir)
    plt.close(fig)
    return fig


def generate_summary_report(
    df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_dir: Path
) -> str:
    """
    Generate a text report summarizing validation results.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SYNTHETICHEALTHSIMULATOR - BIOMARKER VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    lines.append("REFERENCE SOURCES:")
    lines.append("-" * 40)
    lines.append("• NHANES 2017-2020 (CDC) - US National Health and Nutrition Examination Survey")
    lines.append("• Framingham Heart Study - Long-term cardiovascular cohort")
    lines.append("• ARIC Study - Atherosclerosis Risk in Communities")
    lines.append("• WHO Global Health Observatory")
    lines.append("")

    lines.append("COMPARISON WITH REFERENCE (YEAR 19):")
    lines.append("-" * 40)
    if not comparison_df.empty:
        for _, row in comparison_df.iterrows():
            status = "✓ IN RANGE" if row['in_range'] else "✗ OUT OF RANGE"
            dev_pct = (row['synthetic_mean'] - row['reference_mean']) / row['reference_mean'] * 100
            lines.append(
                f"{row['marker']:20s} | synthetic: {row['synthetic_mean']:6.1f} | "
                f"reference: {row['reference_mean']:6.1f} | "
                f"deviation: {dev_pct:+5.1f}% | {status}"
            )
    else:
        lines.append("No comparison data available.")
    lines.append("")

    lines.append("20-YEAR TRENDS:")
    lines.append("-" * 40)
    for marker in ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']:
        marker_data = _prepare_marker_data(df, marker)
        if not marker_data.empty and len(marker_data) > 1:
            delta = marker_data.iloc[-1]['mean'] - marker_data.iloc[0]['mean']
            ref_delta = getattr(DEFAULT_COEFFICIENTS.ref, marker, {}).get('delta_20yr', {}).get('mean', 0)
            lines.append(f"{marker:20s} | Δ = {delta:+7.1f} | reference Δ = {ref_delta:+7.1f}")
    lines.append("")

    lines.append("SAVED FILES:")
    lines.append("-" * 40)
    for f in sorted(output_dir.glob('*')):
        size_kb = f.stat().st_size / 1024
        lines.append(f"  • {f.name:40s} ({size_kb:7.1f} KB)")
    lines.append("")

    lines.append("=" * 80)
    lines.append("IMPORTANT: All data are synthetic and NOT for clinical use.")
    lines.append("=" * 80)

    report = "\n".join(lines)
    report_path = output_dir / 'validation_summary_report.txt'
    report_path.write_text(report, encoding='utf-8')
    logger.info(f"Saved summary report to {report_path}")

    return report

# Legacy plotting functions for notebooks (kept for compatibility)
def create_histogram(data, title, xlabel, ylabel='Frequency', bins=30, figsize=(10,6), color='steelblue', alpha=0.7, **kwargs):
    """Create and return histogram figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, edgecolor='black', alpha=alpha, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig

def create_scatter(x, y, title, xlabel, ylabel, figsize=(10,6), alpha=0.5, color='steelblue', **kwargs):
    """Create and return scatter plot figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=alpha, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig

def create_heatmap(data, title, annot=True, fmt='.2f', cmap='coolwarm', figsize=(12,10), **kwargs):
    """Create and return heatmap figure."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, ax=ax, **kwargs)
    ax.set_title(title)
    return fig

def create_bar_plot(categories, values, title, xlabel, ylabel, figsize=(10,6), color='steelblue', rotation=0, **kwargs):
    """Create and return bar plot figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(categories, values, color=color, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(categories, rotation=rotation)
    ax.grid(True, alpha=0.3, axis='y')
    return fig

def create_line_plot(x, y, title, xlabel, ylabel, figsize=(10,6), color='steelblue', marker='o', **kwargs):
    """Create and return line plot figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, color=color, marker=marker, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig

def create_roc_curve(y_true, y_scores, title='ROC Curves', figsize=(8,6), output_path=None):
    """Create ROC curves for multiple models."""
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0,1],[0,1],'k--',label='Random guess',linewidth=1.5)
    colors = plt.cm.tab10.colors
    for i,(name,scores) in enumerate(y_scores.items()):
        fpr,tpr,_ = roc_curve(y_true,scores)
        auc = roc_auc_score(y_true,scores)
        color = colors[i%len(colors)]
        ax.plot(fpr,tpr,color=color,linewidth=2,label=f'{name} (AUC={auc:.3f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right',fontsize=9)
    ax.grid(True,alpha=0.3)
    if output_path:
        save_figure(fig,output_path,output_dir=Path(output_path).parent)
    return fig

def create_precision_recall_curve(y_true, y_scores, title='Precision-Recall Curves', figsize=(8,6), output_path=None):
    """Create precision-recall curves for multiple models."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    for i,(name,scores) in enumerate(y_scores.items()):
        precision,recall,_ = precision_recall_curve(y_true,scores)
        ap = average_precision_score(y_true,scores)
        color = colors[i%len(colors)]
        ax.plot(recall,precision,color=color,linewidth=2,label=f'{name} (AP={ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='best',fontsize=9)
    ax.grid(True,alpha=0.3)
    if output_path:
        save_figure(fig,output_path,output_dir=Path(output_path).parent)
    return fig

def create_calibration_curve(y_true, y_scores, n_bins=10, title='Calibration Curves', figsize=(8,6), output_path=None):
    """Create calibration curves for multiple models."""
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    for i,(name,scores) in enumerate(y_scores.items()):
        prob_true,prob_pred = calibration_curve(y_true,scores,n_bins=n_bins)
        color = colors[i%len(colors)]
        ax.plot(prob_pred,prob_true,'s-',color=color,label=name,markersize=6,linewidth=1.5)
    ax.plot([0,1],[0,1],'k--',label='Perfect calibration',linewidth=1.5)
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='best',fontsize=9)
    ax.grid(True,alpha=0.3)
    if output_path:
        save_figure(fig,output_path,output_dir=Path(output_path).parent)
    return fig

def create_feature_importance_plot(importances, title='Feature Importance', top_n=15, figsize=(10,8), output_path=None, color='steelblue'):
    """Create horizontal bar plot of feature importances."""
    fig, ax = plt.subplots(figsize=figsize)
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n])
    y_pos = range(len(sorted_imp))
    ax.barh(y_pos, list(sorted_imp.values()), color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(sorted_imp.keys()), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    if output_path:
        save_figure(fig,output_path,output_dir=Path(output_path).parent)
    return fig
