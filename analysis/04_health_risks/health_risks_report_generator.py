#!/usr/bin/env python3
"""
health_risks_report_generator.py — создаёт Markdown-отчёт для анализа рисков.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.report_generator import build_report


def pivot_risk_by_sex(df):
    """Преобразует таблицу risk_means_by_sex в широкий формат."""
    if "Group" in df.columns and "Risk" in df.columns and "Mean" in df.columns:
        return df.pivot(index="Group", columns="Risk", values="Mean").round(4)
    return df


sections = [
    {"title": "# Отчёт по анализу рисков (health_risks)", "type": "text", "text": ""},
    {"title": "## Краткие выводы", "type": "summary"},

    {"title": "## Общая статистика рисков", "type": "table", "table_name": "risk_summary_stats",
     "caption": "Средние риски и распространённость"},

    {"title": "## Распределения рисков (процентили)", "type": "table", "table_name": "risk_percentiles",
     "caption": "5, 25, 50, 75, 95 процентили"},

    {"title": "## Валидация формул рисков", "type": "table", "table_name": "risk_validation_check",
     "caption": "Проверка диапазонов и согласованности"},

    {"title": "## Калибровка моделей риска", "type": "table", "table_name": "calibration_metrics",
     "caption": "Brier score, E/O ratio, slope"},

    {"title": "## Средние риски по полу", "type": "table", "table_name": "risk_means_by_sex",
     "transform": pivot_risk_by_sex, "caption": "Средние значения"},

    {"title": "## Корреляции рисков с клиническими признаками", "type": "table",
     "table_name": "risk_feature_correlations",
     "caption": "Попарные корреляции"},

    {"title": "## Сравнение моделей для CVD", "type": "table", "table_name": "cvd_model_comparison",
     "caption": "AUC-ROC"},

    {"title": "### Важность признаков (Random Forest)", "type": "table", "table_name": "cvd_feature_importance",
     "caption": "Топ-10 признаков",
     "transform": lambda df: df.sort_values("Importance", ascending=False).head(
         10) if "Importance" in df.columns else df},

    {"title": "## Прогнозирование всех заболеваний (Random Forest)", "type": "table", "table_name": "all_diseases_auc",
     "caption": "AUC-ROC и средняя точность"},
]

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    report_path = output_dir / ("report_health_risks.md")
    print(f"Generating report from {output_dir} ...")
    build_report(output_dir, report_path, sections)
    print(f"Report saved to {report_path}")
