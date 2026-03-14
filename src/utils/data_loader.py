"""
Standardized data loading functions for all 5 synthetic datasets.
"""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_baseline(data_path: Path, version: str) -> pd.DataFrame:
    filepath = data_path / f"01_cohort_baseline_{version}.csv"
    return pd.read_csv(filepath)


def load_lifestyle(data_path: Path, version: str) -> pd.DataFrame:
    filepath = data_path / f"02_lifestyle_history_{version}.csv"
    return pd.read_csv(filepath)


def load_biomarkers(data_path: Path, version: str) -> pd.DataFrame:
    filepath = data_path / f"03_biomarkers_history_{version}.csv"
    return pd.read_csv(filepath)


def load_risks(data_path: Path, version: str) -> pd.DataFrame:
    filepath = data_path / f"04_health_risks_{version}.csv"
    return pd.read_csv(filepath)


def load_aggregated(data_path: Path, version: str) -> pd.DataFrame:
    filepath = data_path / f"05_aggregated_dataset_with_missing_{version}.csv"
    return pd.read_csv(filepath)


def load_all_datasets(
        data_path: Path,
        version: str,
        datasets: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    if datasets is None:
        datasets = ['baseline', 'lifestyle', 'biomarkers', 'risks', 'aggregated']

    loaders = {
        'baseline': load_baseline,
        'lifestyle': load_lifestyle,
        'biomarkers': load_biomarkers,
        'risks': load_risks,
        'aggregated': load_aggregated,
    }

    loaded = {}
    for name in datasets:
        if name in loaders:
            loaded[name] = loaders[name](data_path, version)
            print(f"  Loaded {name}: {loaded[name].shape}")

    return loaded


def verify_key_columns(df: pd.DataFrame, required_cols: list, df_name: str = "Dataset") -> bool:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f" ! {df_name}: Missing columns: {missing}")
        return False
    print(f"  {df_name}: All {len(required_cols)} key columns present")
    return True


def load_merged_dataset(step_name: str, data_path: Path, version: str) -> pd.DataFrame:
    """
    Загружает и объединяет датасеты для конкретного шага анализа.
    Упрощает код ноутбуков, убирая дублирующиеся merge операции.
    
    Args:
        step_name: Имя шага ('01_cohort_baseline', '02_lifestyle_history', и т.д.)
        data_path: Путь к данным.
        version: Версия данных.
        
    Returns:
        pd.DataFrame: Объединенный датасет для анализа.
    """
    # Загружаем все базовые датасеты
    baseline = load_baseline(data_path, version)
    lifestyle = load_lifestyle(data_path, version)
    biomarkers = load_biomarkers(data_path, version)
    risks = load_risks(data_path, version)

    if step_name == '01_cohort_baseline':
        # Для когорты часто нужно сравнить с рисками
        return baseline.merge(risks[['person_id', 'has_diabetes', 'has_cvd', 'has_stroke']], on='person_id', how='left')

    elif step_name == '02_lifestyle_history':
        # Список всех факторов, которые нужно преобразовать в длинный формат
        stubnames = [
            'alcohol_g_per_week',
            'cardio_met_minutes',
            'strength_met_minutes',
            'cigarettes_per_day',
            'stress_level',
            'sleep_hours',
            'weight_kg_true',
            'muscle_mass_factor_true',
            'age',
            'cumulative_smoking',
            'fiber_g_day',
            'saturated_fat_pct',
            'macro_imbalance',
            'total_met_minutes'
        ]

        # Преобразование из широкого в длинный формат
        long_df = pd.wide_to_long(
            lifestyle,
            stubnames=stubnames,
            i='person_id',
            j='year',
            sep='_',
            suffix=r'\d+'
        ).reset_index()

        # Добавляем демографические и генетические данные
        long_df = long_df.merge(
            baseline[['person_id', 'sex', 'age_start', 'genetic_risk_diabetes', 'genetic_risk_cvd']],
            on='person_id',
            how='left'
        )

        # Добавляем исходы заболеваний
        long_df = long_df.merge(
            risks[['person_id', 'has_diabetes', 'has_cvd']],
            on='person_id',
            how='left'
        )

        return long_df

    elif step_name == '03_biomarkers':
        # Преобразуем biomarkers в long формат
        long_bio = pd.wide_to_long(
            biomarkers,
            stubnames=['hdl_mgdl', 'total_cholesterol_mgdl', 'sbp_mmhg',
                       'hba1c_percent', 'bmi', 'weight_kg', 'muscle_mass_factor', 'age'],
            i='person_id', j='year', sep='_', suffix=r'\d+'
        ).reset_index()

        # Добавляем демографию и исходы
        long_bio = long_bio.merge(baseline[['person_id', 'sex', 'age_start']], on='person_id')
        long_bio = long_bio.merge(risks[['person_id', 'has_diabetes', 'has_cvd']], on='person_id')
        return long_bio

    elif step_name == '04_health_risks':
        # Для рисков нужен полный контекст: исходные данные + финальные биомаркеры + агрегация
        df = risks.merge(baseline[['person_id', 'age_start', 'height_cm', 'weight_start_kg']], on='person_id',
                         how='left')

        # Добавляем агрегированные данные (финальные значения)
        agg = load_aggregated(data_path, version)
        cols_to_merge = ['person_id', 'final_bmi', 'final_hdl_mgdl', 'final_sbp_mmhg',
                         'final_hba1c_percent', 'avg_alcohol_g_per_week', 'avg_total_met_minutes',
                         'cumulative_smoking_end']
        # Фильтруем колонки, которые могут отсутствовать
        cols_exist = [c for c in cols_to_merge if c in agg.columns]
        df = df.merge(agg[cols_exist], on='person_id', how='left')
        return df

    elif step_name == '05_aggregated_pipeline':
        # Возвращаем готовый агрегированный датасет
        return load_aggregated(data_path, version)

    else:
        raise ValueError(f"Unknown step name: {step_name}")
