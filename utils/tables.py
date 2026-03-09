"""
Standardized table export functions for analysis notebooks.
"""
from pathlib import Path
from typing import Optional, List

import pandas as pd


def save_table(
        df: pd.DataFrame,
        filename: str,
        output_dir: str = "output",
        index: bool = False,
        float_format: str = "%.4f",
        verbose: bool = True
) -> str:
    """
    Save DataFrame as CSV with standard settings.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    df.to_csv(filepath, index=index, float_format=float_format, encoding='utf-8')
    print()

    if verbose:
        print(f"Saved: {filepath} ({df.shape[0]} rows × {df.shape[1]} cols)")
    return str(filepath)


def print_markdown_table(
        df: pd.DataFrame,
        title: str,
        round_cols: Optional[List] = None,
        decimals: int = 2,
        index: bool = False
):
    """
    Print DataFrame as markdown table with title.
    """
    print(f"\n### {title}")

    if round_cols:
        df_display = df.copy()
        for col in round_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(decimals)
    else:
        df_display = df

    markdown_output = df_display.to_markdown(index=index)
    print(markdown_output)
