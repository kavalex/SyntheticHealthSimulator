#!/usr/bin/env python3
"""
cohort_baseline_report_generator.py — создаёт Markdown-отчёт для анализа когорты.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.report_generator import build_report

# Определяем секции
sections = [
    {"title": "# Отчёт по анализу когорты (cohort_baseline)", "type": "text", "text": ""},
    {"title": "## Краткие выводы", "type": "summary"},
    {"title": "## Описательная статистика", "type": "table", "table_name": "baseline_summary",
     "caption": "Основные признаки"},
    {"title": "## Распределения признаков (процентили)", "type": "table", "table_name": "baseline_percentiles",
     "caption": "5, 25, 50, 75, 95 процентили"},
    {"title": "## Пол", "type": "table", "table_name": "sex_distribution",
     "caption": "Количество и доля"},
    {"title": "## Сравнение по полу (средние ± std)", "type": "table", "table_name": "baseline_summary_by_sex"},
    {"title": "## Ключевые корреляции", "type": "table", "table_name": "key_correlations"},
    {"title": "## Предсказание заболеваний по исходным данным", "type": "table", "table_name": "prediction_results",
     "caption": "AUC-ROC для логистической регрессии и случайного леса"},
    {"title": "## Важность признаков (диабет)", "type": "table", "table_name": "feature_importance_diabetes",
     "caption": "Random Forest"},
]

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    report_path = output_dir / "report_cohort_baseline.md"
    print(f"Generating report from {output_dir} ...")
    build_report(output_dir, report_path, sections)
    print(f"Report saved to {report_path}")
