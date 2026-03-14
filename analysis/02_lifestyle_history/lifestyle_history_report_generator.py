#!/usr/bin/env python3
"""
generate_report.py — создаёт Markdown-отчёт для анализа образа жизни.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.report_generator import build_report, generate_mermaid_line_chart


# Функция для создания Mermaid-графиков трендов
def mermaid_trends(tables, output_dir):
    """Генерирует Mermaid-код для всех факторов."""
    if "lifestyle_trends_by_year" not in tables:
        return ""

    df = tables["lifestyle_trends_by_year"]
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if not mean_cols:
        return ""

    mermaid_blocks = []
    for col in mean_cols:
        factor = col.replace("_mean", "")
        df_m = df[["year", col]].rename(columns={col: "value"})
        mermaid = generate_mermaid_line_chart(df_m, "year", "value", f"Тренд {factor}", years=[0, 5, 10, 15, 19])
        mermaid_blocks.append(mermaid)
    return "\n\n".join(mermaid_blocks)


def pivot_by_sex(df):
    """Преобразует таблицу трендов по полу в широкий формат."""
    return (df.melt(id_vars=["year", "sex"], var_name="factor", value_name="value")
            .pivot_table(index=["year", "factor"], columns="sex", values="value")
            .reset_index())


def pivot_by_diabetes(df):
    """Преобразует таблицу трендов по диабету в широкий формат."""
    return (df.melt(id_vars=["year", "has_diabetes"], var_name="factor", value_name="value")
            .pivot_table(index=["year", "factor"], columns="has_diabetes", values="value")
            .reset_index())


sections = [
    {"title": "# Отчёт по анализу образа жизни (lifestyle_history)", "type": "text", "text": ""},
    {"title": "## Краткие выводы", "type": "summary"},

    {"title": "## Распределения факторов (процентили)", "type": "table", "table_name": "lifestyle_percentiles",
     "caption": "5, 25, 50, 75, 95 процентили"},

    {"title": "## Тренды средних значений", "type": "custom",
     "custom_func": lambda tables, output_dir: (
         tables["lifestyle_trends_by_year"][
             ["year"] + [c for c in tables["lifestyle_trends_by_year"].columns if c.endswith("_mean")]]
         .set_index("year").round(2).to_markdown()
     ) if "lifestyle_trends_by_year" in tables else None,
     "caption": "Средние значения по годам"},

    {"title": "### Mermaid-графики трендов", "type": "mermaid", "mermaid_func": mermaid_trends},

    {"title": "## Оценка параметров OU-процесса", "type": "table", "table_name": "ou_theta_estimates",
     "caption": "Оценка θ (скорость возврата)"},

    {"title": "## Тренды по полу", "type": "table", "table_name": "lifestyle_trends_by_sex",
     "transform": pivot_by_sex,
     "caption": "Средние значения по полу"},

    {"title": "## Тренды по статусу диабета", "type": "table", "table_name": "lifestyle_trends_by_has_diabetes",
     "transform": pivot_by_diabetes,
     "caption": "Средние значения по статусу диабета"},

    {"title": "## Корреляции между факторами", "type": "table", "table_name": "lifestyle_correlations",
     "caption": "Попарные корреляции по годам"},

    {"title": "## Сводка по кластерам", "type": "table", "table_name": "cluster_summary",
     "caption": "Характеристики кластеров"},

    {"title": "## Прогнозирование будущего по первым 10 годам", "type": "table",
     "table_name": "lifestyle_prediction_results",
     "caption": "R² и RMSE для веса и стресса"},
]

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    report_path = output_dir / "report_lifestyle_history.md"
    print(f"Generating report from {output_dir} ...")
    build_report(output_dir, report_path, sections)
    print(f"Report saved to {report_path}")
