#!/usr/bin/env python3
"""
biomarkers_report_generator.py — создаёт Markdown-отчёт для анализа биомаркеров.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.report_generator import build_report, generate_mermaid_line_chart


def mermaid_biomarker_trends(tables, output_dir):
    """Генерирует Mermaid-графики для трендов биомаркеров."""
    if "biomarker_trends_by_year" not in tables:
        return ""

    df = tables["biomarker_trends_by_year"]
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if not mean_cols:
        return ""

    mermaid_blocks = []
    for col in mean_cols:
        marker = col.replace("_mean", "")
        df_m = df[["year", col]].rename(columns={col: "value"})
        mermaid = generate_mermaid_line_chart(df_m, "year", "value", f"Тренд {marker}", years=[0, 5, 10, 15, 19])
        mermaid_blocks.append(mermaid)
    return "\n\n".join(mermaid_blocks)


sections = [
    {"title": "# Отчёт по анализу биомаркеров (biomarkers_history)", "type": "text", "text": ""},
    {"title": "## Краткие выводы", "type": "summary"},

    {"title": "## Распределения биомаркеров (процентили)", "type": "table", "table_name": "biomarker_percentiles",
     "caption": "5, 25, 50, 75, 95 процентили"},

    {"title": "## Тренды средних значений", "type": "custom",
     "custom_func": lambda tables, output_dir: (
         tables["biomarker_trends_by_year"][
             ["year"] + [c for c in tables["biomarker_trends_by_year"].columns if c.endswith("_mean")]]
         .set_index("year").round(2).to_markdown()
     ) if "biomarker_trends_by_year" in tables else None,
     "caption": "Средние значения по годам"},

    {"title": "### Mermaid-графики трендов", "type": "mermaid", "mermaid_func": mermaid_biomarker_trends},

    {"title": "## Тренды по полу", "type": "table", "table_name": "biomarker_trends_by_sex",
     "transform": lambda df: (
         df.melt(id_vars=["year", "group"], var_name="marker", value_name="value")
         .pivot_table(index=["year", "marker"], columns="group", values="value")
         .reset_index()
         if set(df.columns) - {"year", "group"} else df
     ), "caption": "Средние значения по полу"},

    {"title": "## Тренды по статусу диабета", "type": "table", "table_name": "biomarker_trends_by_has_diabetes",
     "transform": lambda df: (
         df.melt(id_vars=["year", "group"], var_name="marker", value_name="value")
         .pivot_table(index=["year", "marker"], columns="group", values="value")
         .reset_index()
         if set(df.columns) - {"year", "group"} else df
     ), "caption": "Средние значения по статусу диабета"},

    {"title": "## Корреляции между биомаркерами", "type": "table", "table_name": "biomarker_correlations",
     "caption": "Попарные корреляции по годам"},

    {"title": "## Оценка шума измерений", "type": "table", "table_name": "noise_estimates_with_ci",
     "caption": "Оценка σ с 95% CI"},

    {"title": "## Эффективность сглаживания веса", "type": "table", "table_name": "denoising_summary",
     "caption": "MAE до и после сглаживания (описательная статистика)"},

    {"title": "## Прогнозирование (год 15 по 5-летней истории)", "type": "table", "table_name": "prediction_results",
     "caption": "Сравнение наивного метода и Random Forest"},

    {"title": "### MAE по квартилям истинных значений", "type": "table", "table_name": "prediction_by_quantile",
     "caption": "Ошибка модели в зависимости от уровня маркера",
     "condition": lambda _, tables: "prediction_by_quantile" in tables},
]

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    report_path = output_dir / "report_biomarkers.md"
    print(f"Generating report from {output_dir} ...")
    build_report(output_dir, report_path, sections)
    print(f"Report saved to {report_path}")
