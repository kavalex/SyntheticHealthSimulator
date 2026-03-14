"""
report_generator.py — универсальный модуль для генерации Markdown-отчётов
из сохранённых CSV и текстовых файлов анализа.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd


def load_all_tables(output_dir: Path) -> Dict[str, pd.DataFrame]:
    """Загружает все таблицы из папки tables/ в словарь по имени файла (без расширения)."""
    tables = {}

    # Check for tables subdirectory (new structure)
    tables_dir = output_dir / "tables"
    search_dir = tables_dir if tables_dir.exists() else output_dir

    if not search_dir.exists():
        return tables

    for csv_file in search_dir.glob("*.csv"):
        name = csv_file.stem
        tables[name] = pd.read_csv(csv_file)
    return tables


def df_to_markdown(df: pd.DataFrame, caption: Optional[str] = None, index: bool = False) -> str:
    """Преобразует DataFrame в Markdown-таблицу с опциональным заголовком."""
    md = df.to_markdown(index=index)
    if caption:
        md = f"**{caption}**\n\n{md}"
    return md


def generate_mermaid_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                                years: Optional[List[int]] = None) -> str:
    """
    Генерирует Mermaid-код для линейного графика (graph LR) из данных.
    """
    if df.empty:
        return f"<!-- No data for {title} -->"

    # Filter if years provided
    if years is not None and x_col in df.columns:
        df = df[df[x_col].isin(years)]

    nodes = []
    edges = []
    prev = None
    # Ensure x_col is sorted
    df = df.sort_values(by=x_col)

    for _, row in df.iterrows():
        node_id = f"{x_col}_{row[x_col]}"
        val = row[y_col]
        if isinstance(val, float):
            val_str = f"{val:.1f}"
        else:
            val_str = str(val)
        nodes.append(f"    {node_id}[{row[x_col]}: {val_str}]")
        if prev is not None:
            edges.append(f"    {prev} --> {node_id}")
        prev = node_id
    mermaid = f"```mermaid\ngraph LR\n" + "\n".join(nodes + edges) + "\n```"
    if title:
        mermaid = f"### {title}\n\n{mermaid}"
    return mermaid


def build_report(output_dir: Path, report_path: Path, sections: List[Dict[str, Any]]) -> None:
    """
    Собирает отчёт в Markdown из всех доступных таблиц и summary.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tables = load_all_tables(output_dir)

    report_lines = []

    for section in sections:
        # Проверка условия
        if section.get("condition") and not section["condition"](output_dir, tables):
            continue

        title = section.get("title")
        if title:
            report_lines.append(title)
            report_lines.append("")

        typ = section.get("type", "text")

        if typ == "summary":
            summary_file = output_dir / "analysis_summary.txt"
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    summary = f.read()
                report_lines.append(summary)
                report_lines.append("")
            else:
                report_lines.append("<!-- analysis_summary.txt not found -->")
                report_lines.append("")

        elif typ == "text":
            text = section.get("text", "")
            if text:
                report_lines.append(text)
                report_lines.append("")

        elif typ == "table":
            table_name = section.get("table_name")
            if not table_name or table_name not in tables:
                continue
            df = tables[table_name]
            if "transform" in section:
                df = section["transform"](df)
            report_lines.append(df_to_markdown(df, caption=section.get("caption"), index=section.get("index", False)))
            report_lines.append("")

        elif typ == "mermaid":
            mermaid_func = section.get("mermaid_func")
            if mermaid_func is None:
                continue
            mermaid_code = mermaid_func(tables, output_dir)
            if mermaid_code:
                report_lines.append(mermaid_code)
                report_lines.append("")

        elif typ == "custom":
            custom_func = section.get("custom_func")
            if custom_func:
                content = custom_func(tables, output_dir)
                if content:
                    report_lines.append(content)
                    report_lines.append("")

    report_lines.append("---\n*Отчёт сгенерирован автоматически.*")

    with open(report_path, "w") as f:
        f.write("\n\n".join(report_lines))
