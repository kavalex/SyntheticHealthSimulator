#!/usr/bin/env python3
"""
SyntheticHealthSimulator - Biomarker Statistics Visualization
Comprehensive visualization of all biomarker trends with reference values from NHANES, CDC, WHO.
"""

# %% [markdown]
# # Визуализация статистик биомаркеров
# ## SyntheticHealthSimulator v2.0.0
#
# Этот ноутбук создаёт 8 типов визуализаций на основе файла `all_datasets_statistics.csv`:
# 1. Временные тренды биомаркеров (Line Plot)
# 2. Эволюция распределений (Box Plot по годам)
# 3. Процентильные ленты (Percentile Bands)
# 4. Изменения между годами (Delta Distribution)
# 5. Тепловая карта корреляций (Correlation Heatmap)
# 6. Сводная панель (Dashboard Grid)
# 7. Отклонение от референса (Reference Comparison)
# 8. Сравнение по полу (Sex Comparison)
#
# **Референсные значения из научных источников:**
# - NHANES 2017-2020 (CDC)
# - Framingham Heart Study
# - ARIC Study
# - WHO Global Health Observatory
#
# ⚠️ **ВАЖНО:** Все данные синтетические и НЕ предназначены для клинического использования.

# %% [markdown]
# ## Setup

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Настройки
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Пути
PROJECT_ROOT = Path('/Users/alexeykaverzin/Documents/otus/project/SyntheticHealthSimulator')
STATS_FILE = PROJECT_ROOT / 'output' / 'statistics' / 'all_datasets_statistics.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Statistics file: {STATS_FILE}")
print(f"Output directory: {OUTPUT_DIR}")

# %% [markdown]
# ## Референсные значения из научных источников

# %%
# Референсные значения из NHANES, CDC, WHO, Framingham
REFERENCE_VALUES = {
    'sbp_mmhg': {
        'name': 'Систолическое АД (мм рт.ст.)',
        'source': 'NHANES 2017-2020, CDC',
        'age_40_49': {'mean': 122, 'std': 15, 'p5': 98, 'p95': 148},
        'age_50_59': {'mean': 128, 'std': 16, 'p5': 102, 'p95': 155},
        'age_60_69': {'mean': 135, 'std': 17, 'p5': 108, 'p95': 163},
        'delta_20yr': {'mean': 15, 'min': 10, 'max': 20},  # Framingham
        'unit': 'mmHg'
    },
    'hba1c_percent': {
        'name': 'Гликированный гемоглобин (%)',
        'source': 'CDC Diabetes Report 2020',
        'age_40_49': {'mean': 5.6, 'std': 0.4, 'p5': 5.0, 'p95': 6.3},
        'age_50_59': {'mean': 5.7, 'std': 0.5, 'p5': 5.0, 'p95': 6.5},
        'age_60_69': {'mean': 5.8, 'std': 0.5, 'p5': 5.1, 'p95': 6.6},
        'delta_20yr': {'mean': 0.4, 'min': 0.2, 'max': 0.6},  # ARIC Study
        'unit': '%'
    },
    'hdl_mgdl': {
        'name': 'ЛПВП холестерин (мг/дЛ)',
        'source': 'NHANES 2017-2020',
        'male': {'mean': 45, 'std': 12, 'p5': 28, 'p95': 68},
        'female': {'mean': 58, 'std': 14, 'p5': 38, 'p95': 85},
        'delta_20yr': {'mean': -7, 'min': -10, 'max': -2},  # Framingham
        'unit': 'mg/dL'
    },
    'total_cholesterol_mgdl': {
        'name': 'Общий холестерин (мг/дЛ)',
        'source': 'NHANES 2017-2020',
        'age_40_49': {'mean': 205, 'std': 35, 'p5': 150, 'p95': 265},
        'age_50_59': {'mean': 210, 'std': 38, 'p5': 155, 'p95': 275},
        'age_60_69': {'mean': 200, 'std': 40, 'p5': 145, 'p95': 270},  # statin effect
        'delta_20yr': {'mean': 0, 'min': -10, 'max': 10},
        'unit': 'mg/dL'
    },
    'bmi': {
        'name': 'Индекс массы тела (кг/м²)',
        'source': 'NHANES 2017-2020, CDC',
        'age_40_49': {'mean': 29.0, 'std': 5.5, 'p5': 21.5, 'p95': 39.0},
        'age_50_59': {'mean': 29.5, 'std': 5.8, 'p5': 21.8, 'p95': 40.2},
        'age_60_69': {'mean': 28.8, 'std': 5.5, 'p5': 21.2, 'p95': 39.5},
        'delta_20yr': {'mean': 2.0, 'min': 0.5, 'max': 4.0},  # NHANES longitudinal
        'unit': 'kg/m²'
    },
    'weight_kg': {
        'name': 'Вес (кг)',
        'source': 'NHANES 2017-2020',
        'male': {'mean': 90, 'std': 18, 'p5': 65, 'p95': 125},
        'female': {'mean': 77, 'std': 16, 'p5': 55, 'p95': 108},
        'delta_20yr': {'mean': 3, 'min': 0, 'max': 8},
        'unit': 'kg'
    }
}

# Отображение для графиков
MARKER_DISPLAY_NAMES = {
    'sbp_mmhg': 'SBP (мм рт.ст.)',
    'hba1c_percent': 'HbA1c (%)',
    'hdl_mgdl': 'HDL (мг/дЛ)',
    'total_cholesterol_mgdl': 'TC (мг/дЛ)',
    'bmi': 'BMI (кг/м²)',
    'weight_kg': 'Вес (кг)'
}

print("✅ Референсные значения загружены:")
for marker, ref in REFERENCE_VALUES.items():
    print(f"  • {marker}: {ref['name']} ({ref['source']})")

# %% [markdown]
# ## Загрузка данных

# %%
# Проверка файла
if not STATS_FILE.exists():
    raise FileNotFoundError(f"Statistics file not found: {STATS_FILE}")

# Загрузка
df_stats = pd.read_csv(STATS_FILE)
print(f"\n📊 Загружено статистик: {len(df_stats)} строк")
print(f"Колонки: {df_stats.columns.tolist()}")

# Фильтрация только для биомаркеров
biomarker_prefixes = ['hdl_mgdl', 'total_cholesterol_mgdl', 'sbp_mmhg', 'hba1c_percent', 'bmi', 'weight_kg']
df_biomarkers = df_stats[df_stats['column'].apply(
    lambda x: any(x.startswith(prefix) for prefix in biomarker_prefixes)
)].copy()

print(f"Биомаркеры: {len(df_biomarkers)} строк")
print(f"Уникальные маркеры: {df_biomarkers['column'].nunique()}")


# Извлечение года из названия колонки
def extract_year(col_name: str) -> int:
    """Извлекает год из названия колонки (например, sbp_mmhg_15 → 15)"""
    parts = col_name.split('_')
    return int(parts[-1]) if parts[-1].isdigit() else -1


df_biomarkers['year'] = df_biomarkers['column'].apply(extract_year)
df_biomarkers = df_biomarkers[df_biomarkers['year'] >= 0]


# Извлечение типа маркера
def extract_marker(col_name: str) -> str:
    """Извлекает тип маркера из названия колонки"""
    for prefix in biomarker_prefixes:
        if col_name.startswith(prefix):
            return prefix
    return 'unknown'


df_biomarkers['marker'] = df_biomarkers['column'].apply(extract_marker)

print(f"Года: {sorted(df_biomarkers['year'].unique())}")
print(f"Маркеры: {df_biomarkers['marker'].unique()}")


# %% [markdown]
# ---
# # 1. Временные тренды биомаркеров (Line Plot)
# ---

# %%
def plot_temporal_trends(df: pd.DataFrame, output_dir: Path):
    """
    График 1: Временные тренды биомаркеров с mean ± std и референсными значениями.
    """
    markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, marker in enumerate(markers):
        ax = axes[idx]
        marker_data = df[df['marker'] == marker].sort_values('year')

        if len(marker_data) == 0:
            continue

        # Основные линии
        ax.plot(marker_data['year'], marker_data['mean'], 'o-',
                linewidth=2, label='Синтетика (mean)', color='#2E86AB')
        ax.fill_between(marker_data['year'],
                        marker_data['mean'] - marker_data['std'],
                        marker_data['mean'] + marker_data['std'],
                        alpha=0.3, label='±1σ', color='#2E86AB')

        # Референсные значения (если есть)
        if marker in REFERENCE_VALUES:
            ref = REFERENCE_VALUES[marker]
            # Показываем диапазон для среднего возраста
            if 'age_50_59' in ref:
                ref_range = ref['age_50_59']
                ax.axhspan(ref_range['p5'], ref_range['p95'],
                           alpha=0.2, color='green', label='Референс (NHANES p5-p95)')
                ax.axhline(ref_range['mean'], linestyle='--', color='green',
                           linewidth=2, label=f'Референс mean={ref_range["mean"]}')

        # Настройки
        ax.set_xlabel('Год наблюдения', fontsize=11)
        ax.set_ylabel(MARKER_DISPLAY_NAMES.get(marker, marker), fontsize=11)
        ax.set_title(f'{ref["name"] if marker in REFERENCE_VALUES else marker}\nТренд за 20 лет',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Расчёт изменения
        if len(marker_data) > 1:
            delta = marker_data.iloc[-1]['mean'] - marker_data.iloc[0]['mean']
            delta_ref = ref['delta_20yr']['mean'] if marker in REFERENCE_VALUES else 0
            ax.text(0.02, 0.98, f'Δ = {delta:+.1f} (реф: {delta_ref:+.1f})',
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    filepath = output_dir / '01_temporal_trends.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig


# %%
plot_temporal_trends(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # 2. Эволюция распределений (Box Plot по годам)
# ---

# %%
def plot_distribution_evolution(df: pd.DataFrame, output_dir: Path):
    """
    График 2: Box plot для ключевых годов (0, 5, 10, 15, 19).
    """
    markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']
    years_to_show = [0, 5, 10, 15, 19]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, marker in enumerate(markers):
        ax = axes[idx]
        marker_data = df[df['marker'] == marker]

        if len(marker_data) == 0:
            continue

        # Подготовка данных для box plot
        plot_data = []
        labels = []
        for year in years_to_show:
            year_data = marker_data[marker_data['year'] == year]
            if len(year_data) > 0:
                # Используем percentiles для имитации box plot
                p25 = year_data['p25'].values[0] if 'p25' in year_data.columns else year_data['mean'].values[0] - \
                                                                                    year_data['std'].values[0]
                p50 = year_data['median'].values[0] if 'median' in year_data.columns else year_data['mean'].values[0]
                p75 = year_data['p75'].values[0] if 'p75' in year_data.columns else year_data['mean'].values[0] + \
                                                                                    year_data['std'].values[0]
                p5 = year_data['p5'].values[0] if 'p5' in year_data.columns else p25 - 1.5 * (p75 - p25)
                p95 = year_data['p95'].values[0] if 'p95' in year_data.columns else p75 + 1.5 * (p75 - p25)

                plot_data.append([p5, p25, p50, p75, p95])
                labels.append(f'Год {year}')

        # Рисуем box plot
        positions = range(len(plot_data))
        for i, (pos, data) in enumerate(zip(positions, plot_data)):
            p5, p25, p50, p75, p95 = data
            # Усы
            ax.plot([pos, pos], [p5, p95], color='#2E86AB', linewidth=1)
            # Box
            ax.plot([pos - 0.15, pos + 0.15], [p25, p25], color='#2E86AB', linewidth=2)
            ax.plot([pos - 0.15, pos + 0.15], [p75, p75], color='#2E86AB', linewidth=2)
            ax.plot([pos - 0.15, pos - 0.15], [p25, p75], color='#2E86AB', linewidth=2)
            ax.plot([pos + 0.15, pos + 0.15], [p25, p75], color='#2E86AB', linewidth=2)
            # Медиана
            ax.plot([pos - 0.2, pos + 0.2], [p50, p50], color='red', linewidth=2)
            # Mean
            ax.plot(pos, df[(df['marker'] == marker) & (df['year'] == pos)]['mean'].values[0] if len(
                df[(df['marker'] == marker) & (df['year'] == pos)]) > 0 else p50,
                    'g^', markersize=8, label='Mean' if i == 0 else '')

        # Референс
        if marker in REFERENCE_VALUES:
            ref = REFERENCE_VALUES[marker]
            if 'age_50_59' in ref:
                ax.axhspan(ref['age_50_59']['p5'], ref['age_50_59']['p95'],
                           alpha=0.2, color='green', label='Референс (p5-p95)')

        ax.set_xlabel('Год', fontsize=11)
        ax.set_ylabel(MARKER_DISPLAY_NAMES.get(marker, marker), fontsize=11)
        ax.set_title(f'{marker}\nЭволюция распределения', fontsize=12, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = output_dir / '02_distribution_evolution.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig


# %%
plot_distribution_evolution(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # 3. Процентильные ленты (Percentile Bands)
# ---

# %%
def plot_percentile_bands(df: pd.DataFrame, output_dir: Path):
    """
    График 3: p5, p25, p50, p75, p95 как заполненные области.
    """
    markers = ['sbp_mmhg', 'hba1c_percent']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, marker in enumerate(markers):
        ax = axes[idx]
        marker_data = df[df['marker'] == marker].sort_values('year')

        if len(marker_data) == 0:
            continue

        years = marker_data['year'].values

        # p50 (медиана)
        ax.plot(years, marker_data['median'], 'k-', linewidth=2.5, label='Медиана (p50)')

        # p25-p75
        ax.fill_between(years, marker_data['p25'], marker_data['p75'],
                        alpha=0.5, color='#2E86AB', label='p25-p75 (50%)')

        # p5-p95
        ax.fill_between(years, marker_data['p5'], marker_data['p95'],
                        alpha=0.3, color='#2E86AB', label='p5-p95 (90%)')

        # Референс
        if marker in REFERENCE_VALUES:
            ref = REFERENCE_VALUES[marker]
            if 'age_50_59' in ref:
                ax.axhspan(ref['age_50_59']['p5'], ref['age_50_59']['p95'],
                           alpha=0.2, color='green', linestyle='--',
                           label='Референс NHANES (p5-p95)')

        ax.set_xlabel('Год наблюдения', fontsize=11)
        ax.set_ylabel(MARKER_DISPLAY_NAMES.get(marker, marker), fontsize=11)
        ax.set_title(f'{ref["name"] if marker in REFERENCE_VALUES else marker}\nПроцентильные ленты',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = output_dir / '03_percentile_bands.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig


# %%
plot_percentile_bands(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # 4. Изменения между годами (Delta Distribution)
# ---

# %%
def plot_delta_distribution(df: pd.DataFrame, output_dir: Path):
    """
    График 4: Гистограмма изменений (Year 19 - Year 0).
    """
    markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, marker in enumerate(markers):
        ax = axes[idx]
        marker_data = df[df['marker'] == marker]

        year_0 = marker_data[marker_data['year'] == 0]
        year_19 = marker_data[marker_data['year'] == 19]

        if len(year_0) == 0 or len(year_19) == 0:
            continue

        # Расчёт изменения
        delta_mean = year_19['mean'].values[0] - year_0['mean'].values[0]
        delta_std = np.sqrt(year_0['std'].values[0] ** 2 + year_19['std'].values[0] ** 2)

        # Референсное изменение
        if marker in REFERENCE_VALUES:
            ref_delta = REFERENCE_VALUES[marker]['delta_20yr']['mean']
            ref_min = REFERENCE_VALUES[marker]['delta_20yr']['min']
            ref_max = REFERENCE_VALUES[marker]['delta_20yr']['max']
        else:
            ref_delta, ref_min, ref_max = 0, -5, 5

        # Гистограмма (имитация через нормальное распределение)
        x = np.linspace(delta_mean - 3 * delta_std, delta_mean + 3 * delta_std, 100)
        y = stats.norm.pdf(x, delta_mean, delta_std)
        ax.fill_between(x, y, alpha=0.5, color='#2E86AB', label='Распределение Δ')
        ax.plot(x, y, color='#2E86AB', linewidth=2)

        # Вертикальные линии
        ax.axvline(delta_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Δ синтетика = {delta_mean:+.1f}')
        ax.axvline(ref_delta, color='green', linestyle='-.', linewidth=2,
                   label=f'Δ референс = {ref_delta:+.1f}')

        # Референсный диапазон
        ax.axvspan(ref_min, ref_max, alpha=0.2, color='green',
                   label=f'Референс диапазон [{ref_min}, {ref_max}]')

        # Статус
        in_range = ref_min <= delta_mean <= ref_max
        status = '✅ OK' if in_range else '❌ WARN'
        ax.text(0.02, 0.98, status, transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if in_range else 'lightcoral', alpha=0.5))

        ax.set_xlabel('Изменение за 20 лет', fontsize=11)
        ax.set_ylabel('Плотность', fontsize=11)
        ax.set_title(f'{MARKER_DISPLAY_NAMES.get(marker, marker)}\nРаспределение изменений',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = output_dir / '04_delta_distribution.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig


# %%
plot_delta_distribution(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # 5. Сводная панель (Dashboard Grid)
# ---

# %%
def plot_dashboard_grid(df: pd.DataFrame, output_dir: Path):
    """
    График 6: 2x3 сетка всех основных биомаркеров (small multiples).
    """
    markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'total_cholesterol_mgdl', 'bmi', 'weight_kg']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, marker in enumerate(markers):
        ax = axes[idx]
        marker_data = df[df['marker'] == marker].sort_values('year')

        if len(marker_data) == 0:
            continue

        # Линия тренда
        ax.plot(marker_data['year'], marker_data['mean'], 'o-',
                linewidth=2, color='#2E86AB', label='Mean')
        ax.fill_between(marker_data['year'],
                        marker_data['mean'] - marker_data['std'],
                        marker_data['mean'] + marker_data['std'],
                        alpha=0.3, color='#2E86AB')

        # Референс
        if marker in REFERENCE_VALUES:
            ref = REFERENCE_VALUES[marker]
            if 'age_50_59' in ref:
                ax.axhline(ref['age_50_59']['mean'], color='green',
                           linestyle='--', linewidth=1.5, alpha=0.7)

        # Статистика
        delta = marker_data.iloc[-1]['mean'] - marker_data.iloc[0]['mean']
        ax.text(0.02, 0.95, f'Δ = {delta:+.1f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Год', fontsize=10)
        ax.set_ylabel(MARKER_DISPLAY_NAMES.get(marker, marker), fontsize=10)
        ax.set_title(marker, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 20)

    # Общий заголовок
    fig.suptitle('SyntheticHealthSimulator — Сводная панель биомаркеров (20 лет)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    filepath = output_dir / '05_dashboard_grid.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig


# %%
plot_dashboard_grid(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # 6. Отклонение от референса (Reference Comparison)
# ---

# %%
def plot_reference_comparison(df: pd.DataFrame, output_dir: Path):
    """
    График 7: Синтетика vs Реальные данные (NHANES) с зонами "нормы".
    """
    markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    comparison_results = []

    for idx, marker in enumerate(markers):
        ax = axes[idx]
        marker_data = df[df['marker'] == marker].sort_values('year')

        if len(marker_data) == 0 or marker not in REFERENCE_VALUES:
            continue

        ref = REFERENCE_VALUES[marker]
        year_19 = marker_data[marker_data['year'] == 19]

        if len(year_19) == 0:
            continue

        syn_mean = year_19['mean'].values[0]
        syn_std = year_19['std'].values[0]

        # Референсные значения
        if 'age_50_59' in ref:
            ref_mean = ref['age_50_59']['mean']
            ref_p5 = ref['age_50_59']['p5']
            ref_p95 = ref['age_50_59']['p95']
        else:
            continue

        # Визуализация
        # Референсный диапазон
        ax.axvspan(ref_p5, ref_p95, alpha=0.3, color='green',
                   label=f'Референс (p5-p95): {ref_p5}-{ref_p95}')
        ax.axvline(ref_mean, color='green', linestyle='--', linewidth=2,
                   label=f'Референс mean: {ref_mean}')

        # Синтетика
        ax.axvline(syn_mean, color='#2E86AB', linestyle='-', linewidth=3,
                   label=f'Синтетика mean: {syn_mean:.1f}')
        ax.axvspan(syn_mean - syn_std, syn_mean + syn_std,
                   alpha=0.3, color='#2E86AB', label=f'Синтетика ±σ: {syn_std:.1f}')

        # t-test (имитация)
        t_stat = (syn_mean - ref_mean) / (syn_std / np.sqrt(5000))
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        # Статус
        in_range = ref_p5 <= syn_mean <= ref_p95
        status = '✅ В норме' if in_range else '❌ Отклонение'
        color = 'lightgreen' if in_range else 'lightcoral'

        ax.text(0.02, 0.98, f'{status}\np ≈ {p_value:.4f}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

        ax.set_xlabel(MARKER_DISPLAY_NAMES.get(marker, marker), fontsize=11)
        ax.set_ylabel('Плотность', fontsize=11)
        ax.set_title(f'{ref["name"]}\nСинтетика vs Референс (Год 19)',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Сохранение результатов
        comparison_results.append({
            'marker': marker,
            'synthetic_mean': syn_mean,
            'synthetic_std': syn_std,
            'reference_mean': ref_mean,
            'reference_p5': ref_p5,
            'reference_p95': ref_p95,
            't_statistic': t_stat,
            'p_value': p_value,
            'in_range': in_range
        })

    # Сохранение таблицы сравнения
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_dir / 'reference_comparison_table.csv', index=False)
    print(f"✅ Сохранено: {output_dir / 'reference_comparison_table.csv'}")

    plt.tight_layout()
    filepath = output_dir / '06_reference_comparison.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig, comparison_df


# %%
plot_reference_comparison(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # 7. Тепловая карта отклонений (Deviation Heatmap)
# ---

# %%
def plot_deviation_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    График 7: Тепловая карта отклонений от референса по годам.
    """
    markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']
    years = list(range(0, 20, 2))  # Каждый 2-й год

    # Подготовка матрицы отклонений
    deviation_matrix = []
    row_labels = []

    for marker in markers:
        marker_data = df[df['marker'] == marker]
        if marker not in REFERENCE_VALUES:
            continue

        ref = REFERENCE_VALUES[marker]
        ref_mean = ref.get('age_50_59', {}).get('mean', None)
        if ref_mean is None:
            continue

        row = []
        for year in years:
            year_data = marker_data[marker_data['year'] == year]
            if len(year_data) > 0:
                syn_mean = year_data['mean'].values[0]
                deviation = (syn_mean - ref_mean) / ref_mean * 100  # Процентное отклонение
                row.append(deviation)
            else:
                row.append(np.nan)

        # ✅ Добавляем только если есть данные
        if len(row) > 0 and any(not np.isnan(v) for v in row):
            deviation_matrix.append(row)
            row_labels.append(MARKER_DISPLAY_NAMES.get(marker, marker))

    # ✅ Проверка: есть ли данные для отображения
    if len(deviation_matrix) == 0:
        print("⚠️ Нет данных для тепловой карты отклонений")
        return None

    deviation_matrix = np.array(deviation_matrix)

    # Тепловая карта
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(deviation_matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)

    # ✅ Подписи — используем фактическую длину матрицы
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f'Год {y}' for y in years])
    ax.set_yticks(range(len(deviation_matrix)))  # ✅ Исправлено: len(deviation_matrix) вместо len(markers)
    ax.set_yticklabels(row_labels)

    # Значения в ячейках
    for i in range(len(deviation_matrix)):
        for j in range(len(years)):
            if not np.isnan(deviation_matrix[i, j]):
                text = ax.text(j, i, f'{deviation_matrix[i, j]:+.1f}%',
                               ha='center', va='center', fontsize=9,
                               color='black' if abs(deviation_matrix[i, j]) < 10 else 'white')

    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Отклонение от референса (%)', fontsize=11)

    ax.set_title('Отклонения синтетических данных от референсных значений (NHANES)\n' +
                 'Зелёный = ниже референса, Красный = выше референса',
                 fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    filepath = output_dir / '07_deviation_heatmap.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig


# %%
plot_deviation_heatmap(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # 8. Сравнение по полу (если доступно)
# ---

# %%
def plot_sex_comparison(df: pd.DataFrame, output_dir: Path):
    """
    График 8: Сравнение биомаркеров по полу (если данные доступны).
    """
    # Проверяем наличие данных по полу
    if 'sex' not in df.columns:
        print("⚠️ Данные по полу отсутствуют в файле статистик")
        print("Создаю заглушку с информацией...")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Данные по полу отсутствуют в all_datasets_statistics.csv\n' +
                'Для анализа по полу необходимо:\n' +
                '1. Модифицировать generate_statistics.py\n' +
                '2. Добавить группировку по колонке "sex"',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        filepath = output_dir / '08_sex_comparison.svg'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    # Если данные есть — строим графики
    markers = ['sbp_mmhg', 'hdl_mgdl', 'bmi']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, marker in enumerate(markers):
        ax = axes[idx]

        for sex in ['M', 'F']:
            sex_data = df[(df['marker'] == marker) & (df['sex'] == sex)].sort_values('year')
            if len(sex_data) > 0:
                ax.plot(sex_data['year'], sex_data['mean'], 'o-',
                        label=f'Пол {sex}', linewidth=2)

        ax.set_xlabel('Год', fontsize=11)
        ax.set_ylabel(MARKER_DISPLAY_NAMES.get(marker, marker), fontsize=11)
        ax.set_title(f'{marker}\nСравнение по полу', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = output_dir / '08_sex_comparison.svg'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Сохранено: {filepath}")
    plt.show()

    return fig


# %%
plot_sex_comparison(df_biomarkers, OUTPUT_DIR)


# %% [markdown]
# ---
# # Сводный отчёт
# ---

# %%
def generate_summary_report(df: pd.DataFrame, comparison_df: pd.DataFrame, output_dir: Path):
    """
    Генерирует текстовый отчёт с ключевыми метриками.
    """
    report_lines = [
        "=" * 80,
        "SYNTHETICHEALTHSIMULATOR — ОТЧЁТ О ВАЛИДАЦИИ БИОМАРКЕРОВ",
        "=" * 80,
        f"Дата генерации: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Источник данных: {STATS_FILE}",
        "",
        "РЕФЕРЕНСНЫЕ ИСТОЧНИКИ:",
        "-" * 40,
        "• NHANES 2017-2020 (CDC) — Национальное обследование здоровья и питания США",
        "• Framingham Heart Study — Долгосрочное сердечно-сосудистое исследование",
        "• ARIC Study — Исследование риска атеросклероза в сообществах",
        "• WHO Global Health Observatory — Глобальная обсерватория здравоохранения ВОЗ",
        "",
        "СРАВНЕНИЕ С РЕФЕРЕНСНЫМИ ЗНАЧЕНИЯМИ (Год 19):",
        "-" * 40,
    ]

    if len(comparison_df) > 0:
        for _, row in comparison_df.iterrows():
            status = "✅ В НОРМЕ" if row['in_range'] else "❌ ОТКЛОНЕНИЕ"
            report_lines.append(
                f"{row['marker']:20s} | Синтетика: {row['synthetic_mean']:6.1f} | "
                f"Референс: {row['reference_mean']:6.1f} | "
                f"Отклонение: {(row['synthetic_mean'] - row['reference_mean']) / row['reference_mean'] * 100:+5.1f}% | {status}"
            )

    report_lines.extend([
        "",
        "ТРЕХЛЕТНИЕ ТРЕНДЫ (Δ за 20 лет):",
        "-" * 40,
    ])

    markers = ['sbp_mmhg', 'hba1c_percent', 'hdl_mgdl', 'bmi']
    for marker in markers:
        marker_data = df[df['marker'] == marker]
        year_0 = marker_data[marker_data['year'] == 0]
        year_19 = marker_data[marker_data['year'] == 19]

        if len(year_0) > 0 and len(year_19) > 0:
            delta = year_19['mean'].values[0] - year_0['mean'].values[0]
            ref_delta = REFERENCE_VALUES.get(marker, {}).get('delta_20yr', {}).get('mean', 0)
            status = "✅" if abs(delta - ref_delta) < 5 else "⚠️"
            report_lines.append(
                f"{marker:20s} | Δ = {delta:+7.1f} | Референс Δ = {ref_delta:+7.1f} | {status}"
            )

    report_lines.extend([
        "",
        "СОХРАНЁННЫЕ ФАЙЛЫ:",
        "-" * 40,
    ])

    for file in sorted(output_dir.glob('*.svg')):
        size_kb = file.stat().st_size / 1024
        report_lines.append(f"  • {file.name:40s} ({size_kb:7.1f} KB)")

    report_lines.extend([
        "",
        "=" * 80,
        "⚠️  ВАЖНО: Все данные синтетические и НЕ предназначены для клинического использования!",
        "=" * 80,
    ])

    # Сохранение отчёта
    report_text = "\n".join(report_lines)
    report_path = output_dir / 'validation_summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✅ Отчёт сохранён: {report_path}")

    return report_text


# %%
# Генерация отчёта (используем comparison_df из предыдущего шага)
# Если comparison_df не определён, создаём пустой
try:
    comparison_df
except NameError:
    comparison_df = pd.DataFrame()

generate_summary_report(df_biomarkers, comparison_df, OUTPUT_DIR)

# %% [markdown]
# ---
# # Выводы
# ---
#
# ## Ключевые метрики валидации
#
# | Биомаркер | Синтетика (Год 19) | Референс (NHANES) | Отклонение | Статус |
# |-----------|-------------------|-------------------|------------|--------|
# | SBP | ~155-170 мм рт.ст. | 128-135 мм рт.ст. | +20-30% | ⚠️ Высокое |
# | HbA1c | ~6.2-6.3% | 5.7-5.8% | +8-10% | ⚠️ Умеренное |
# | HDL | ~43-44 мг/дЛ | 45-58 мг/дЛ | -5-25% | ✅ В норме |
# | BMI | ~27-28 кг/м² | 28.8-29.5 кг/м² | -5% | ✅ В норме |
#
# ## Рекомендации
#
# 1. **SBP (систолическое давление)** — Требуется корректировка коэффициентов:
#    - Снизить `SBPCoefficients.AGE_COEF` с 0.6 до 0.25-0.35
#    - Снизить `SBPCoefficients.CHANGE_MAX` с 5.0 до 2.0-3.0
#
# 2. **HbA1c** — Умеренное завышение тренда:
#    - Снизить `HbA1cCoefficients.BASE_DRIFT` с 0.03 до 0.015
#    - Усилить `FIBER_COEF` и `ACTIVITY_COEF`
#
# 3. **HDL и BMI** — В пределах референсных диапазонов ✅
#
# ## Научные источники
#
# 1. **NHANES 2017-2020**: https://www.cdc.gov/nchs/nhanes/index.htm
# 2. **Framingham Heart Study**: https://www.framinghamheartstudy.org/
# 3. **ARIC Study**: https://aric.cscc.unc.edu/aric9/
# 4. **WHO Global Health Observatory**: https://www.who.int/data/gho
#
# ---
# **Все файлы визуализаций сохранены в:** `output/visualizations/`

# %% [markdown]
# ## Завершение
#
# Все 8 типов визуализаций созданы и сохранены в директорию `output/visualizations/`:
#
# | # | График | Файл |
# |---|--------|------|
# | 1 | Временные тренды | `01_temporal_trends.svg` |
# | 2 | Эволюция распределений | `02_distribution_evolution.svg` |
# | 3 | Процентильные ленты | `03_percentile_bands.svg` |
# | 4 | Изменения между годами | `04_delta_distribution.svg` |
# | 5 | Сводная панель | `05_dashboard_grid.svg` |
# | 6 | Отклонение от референса | `06_reference_comparison.svg` |
# | 7 | Тепловая карта отклонений | `07_deviation_heatmap.svg` |
# | 8 | Сравнение по полу | `08_sex_comparison.svg` |
#
# **Дополнительно:**
# - `reference_comparison_table.csv` — Таблица сравнения с референсом
# - `validation_summary_report.txt` — Текстовый отчёт

# %%
print("\n" + "=" * 80)
print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
print("=" * 80)
print(f"\nВсе файлы сохранены в: {OUTPUT_DIR}")
print("\nСписок файлов:")
for f in sorted(OUTPUT_DIR.glob('*')):
    size = f.stat().st_size / 1024
    print(f"  • {f.name:50s} ({size:8.1f} KB)")
print("\n" + "=" * 80)
