#!/usr/bin/env python3
"""
aggregated_report_generator.py — создаёт Markdown-отчёт для агрегированного датасета.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.report_generator import build_report


def fix_feature_importance(df):
    """Исправляет имена столбцов в feature_importance_aggregated."""
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Feature'})
    df = df.sort_values('importance', ascending=False).head(15)
    return df


sections = [
    {"title": "# Отчёт по агрегированному датасету (aggregated)", "type": "text", "text": ""},
    {"title": "## Краткие выводы", "type": "summary"},

    {"title": "## Распределения ключевых признаков (процентили)", "type": "table",
     "table_name": "aggregated_percentiles",
     "caption": "5, 25, 50, 75, 95 процентили"},

    {"title": "## Анализ пропущенных значений (MAR)", "type": "table", "table_name": "missing_values_report",
     "caption": "Статистика пропусков"},

    {"title": "## Валидация датасета", "type": "table", "table_name": "aggregated_validation_check",
     "caption": "Проверка соответствия спецификации"},

    {"title": "## Целевая переменная (диабет)", "type": "table", "table_name": "target_risk_summary",
     "caption": "Распространённость и риск"},

    {"title": "## Сравнение методов импутации", "type": "table", "table_name": "imputation_comparison"},

    {"title": "## Сконструированные признаки", "type": "table", "table_name": "engineered_features",
     "caption": "Список новых признаков"},

    {"title": "## Подбор моделей (кросс-валидация)", "type": "table", "table_name": "cv_model_selection",
     "caption": "Лучшие параметры и AUC"},

    {"title": "## Сравнение моделей на тестовой выборке", "type": "table", "table_name": "model_comparison_final",
     "caption": "AUC, Avg Precision, Brier"},

    {"title": "## Важность признаков (Random Forest)", "type": "table", "table_name": "feature_importance_aggregated",
     "transform": fix_feature_importance, "caption": "Топ-15 признаков"},

    {"title": "## Анализ порога принятия решений", "type": "table", "table_name": "threshold_analysis",
     "caption": "Precision, Recall, F1 по порогам"},

    {"title": "## Сравнение с рисками (Notebook 4)", "type": "table", "table_name": "notebook_comparison_4_vs_5",
     "caption": "Улучшение от добавления признаков образа жизни и биомаркеров"},
]

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    report_path = output_dir / "report_aggregated.md"
    print(f"Generating report from {output_dir} ...")
    build_report(output_dir, report_path, sections)
    print(f"Report saved to {report_path}")
