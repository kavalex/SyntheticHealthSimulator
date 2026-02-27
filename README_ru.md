# SyntheticHealthSimulator

**Генератор синтетических медицинских данных для задач машинного обучения**

[![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/SyntheticHealthSimulator/releases)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

---

## ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ

Данные, генерируемые этим инструментом, являются полностью синтетическими.  
Они основаны на механистических моделях, упрощениях и калибровках, выполненных специально для обучения моделей машинного обучения.

**НЕ ИСПОЛЬЗУЙТЕ ЭТИ ДАННЫЕ ДЛЯ КЛИНИЧЕСКИХ ДИАГНОЗОВ, ЛЕЧЕНИЯ ИЛИ ЛЮБЫХ ДРУГИХ МЕДИЦИНСКИХ РЕШЕНИЙ.**  
Инструмент предназначен исключительно для образовательных и исследовательских целей в области ML.

---

## О проекте

`SyntheticHealthSimulator` генерирует реалистичные (с точки зрения статистических закономерностей) панельные данные о состоянии здоровья виртуальной популяции на протяжении 20 лет. Каждый «пациент» имеет:

- демографические характеристики (возраст, пол, рост);
- генетический профиль (полигенные риски);
- историю образа жизни (питание, физическая активность, курение, сон, стресс);
- ежегодно измеряемые биомаркеры (вес, давление, холестерин, HbA1c).

На основе этих данных рассчитываются риски развития различных заболеваний (сердечно-сосудистые, диабет 2 типа, инсульт, НАЖБ, колоректальный рак, цирроз) и моделируются исходы.

Основное назначение — создание качественных датасетов для обучения моделей прогнозирования, исследования влияния факторов образа жизни и тестирования алгоритмов обработки пропущенных данных.

---

## Установка и зависимости

Проект написан на **Python 3.14+** и использует стандартные библиотеки: `numpy`, `pandas`.

```bash
# Установка зависимостей (если используется pip)
pip install numpy pandas

# Или с помощью uv (рекомендуется)
uv sync
```

После установки склонируйте репозиторий или скопируйте файл `generator.py`.

---

## Использование

Базовый запуск генерации данных (5000 человек, 20 лет наблюдения):

```bash
python generator.py
```

Все настройки задаются в функции `main()`:

- `N_PEOPLE` — количество синтетических пациентов.
- `YEARS` — период наблюдения (лет).
- `seed` — случайное зерно для воспроизводимости.
- `diabetes_threshold` — порог калибровки риска диабета (по умолчанию 18.7, обеспечивает prevalence ~15%).

После завершения работы в папке `data/synthetic` будут созданы пять CSV-файлов (и соответствующие JSON-файлы с метаданными).

---

## Структура генерируемых данных

Все файлы, кроме агрегированного, имеют панельную структуру: для каждого пациента и года наблюдения хранится одна строка или набор показателей. Ниже приведены колонки для каждого файла.

### 1. Базовая когорта (`01_cohort_baseline.csv`)
Одна строка на пациента. Содержит неизменные характеристики и начальные значения.

- **Идентификатор и демография**: `person_id`, `age_start`, `sex`, `height_cm`
- **Антропометрия**: `weight_start_kg`, `bmi_start`, `muscle_mass_start`
- **Энергетический обмен**: `bmr_start_kcal`, `base_calories`
- **Генетические риски**: `genetic_risk_cvd`, `genetic_risk_diabetes`, `genetic_risk_stroke`, `genetic_risk_nafld`, `genetic_risk_colorectal`, `genetic_risk_cirrhosis`
- **Начальные биомаркеры**: `initial_hdl_mgdl`, `initial_total_cholesterol_mgdl`, `initial_sbp_mmhg`, `initial_hba1c_percent`

### 2. История образа жизни (`02_lifestyle_history.csv`)
Для каждого пациента и каждого года фиксируются показатели образа жизни и расчётные параметры.

- **Идентификатор, год, возраст**: `person_id`, `year`, `age`
- **Физические показатели (истинные, без шума)**: `weight_kg_true`, `muscle_mass_factor_true`, `bmr_kcal_true`, `tdee_kcal_true`
- **Питание**: `protein_pct`, `fat_pct`, `carb_pct`, `saturated_fat_pct`, `simple_carbs_pct`, `fiber_g_day`, `sodium_mg_day`, `kbju_imbalance`, `simple_carbs_g`
- **Образ жизни**: `alcohol_g_per_week`, `cardio_met_minutes`, `strength_met_minutes`, `total_met_minutes`, `cigarettes_per_day`, `stress_level`, `sleep_hours`, `cumulative_smoking`

### 3. Биомаркеры (`03_biomarkers_history.csv`)
Ежегодные значения биомаркеров (с добавленным шумом).

- **Идентификатор, год**: `person_id`, `year`
- **Биомаркеры**: `weight_kg`, `hdl_mgdl`, `total_cholesterol_mgdl`, `sbp_mmhg`, `hba1c_percent`
- **Производные показатели**: `bmi`, `bmi_corrected`, `non_hdl_mgdl`, `non_hdl_mmol`, `muscle_mass_factor`

### 4. Риски и исходы (`04_health_risks.csv`)
Рассчитанные 10-летние риски и бинарные флаги наличия заболеваний за весь период наблюдения.

- **Идентификатор**: `person_id`
- **Риски**: `cvd_risk_10year`, `diabetes_risk_10year`, `stroke_risk_10year`, `nafld_risk_10year`, `colorectal_risk_10year`, `cirrhosis_risk_10year`
- **Исходы**: `has_cvd`, `has_diabetes`, `has_stroke`, `has_nafld`, `has_colorectal`, `has_cirrhosis`
- **Дополнительно**: `health_score`, `primary_death_cause`, `estimated_event_age`, `sex`

### 5. Агрегированный датасет с пропусками (`05_aggregated_dataset_with_missing.csv`)
Одна строка на пациента. Содержит усреднённые и стартовые/конечные значения факторов образа жизни, финальные биомаркеры, риски, исходы и синтезированные пропуски (по механизму MAR). Пропуски могут встречаться в колонках: `final_sbp_mmhg`, `final_hdl_mgdl`, `final_total_cholesterol_mgdl`, `final_hba1c_percent`, `avg_alcohol_g_per_week`, `avg_total_met_minutes`.

- **Идентификатор и демография**: `person_id`, `sex`, `age_start`, `age_end`
- **Усреднённые факторы образа жизни**: `avg_alcohol_g_per_week`, `avg_total_met_minutes`, `avg_sleep_hours`, `avg_stress_level`, `avg_fiber_g_day`, `avg_sodium_mg_day`, `avg_saturated_fat_pct`, `avg_simple_carbs_g`, `avg_kbju_imbalance`
- **Динамика веса и мышечной массы**: `bmi_start`, `bmi_end`, `bmi_change`, `weight_start_kg`, `weight_end_kg`, `weight_change`, `muscle_mass_factor_start`, `muscle_mass_factor_end`, `muscle_mass_factor_change`
- **Финальные биомаркеры**: `final_hdl_mgdl`, `final_total_cholesterol_mgdl`, `final_sbp_mmhg`, `final_hba1c_percent`, `final_non_hdl_mmol`
- **Курение**: `avg_cigarettes_per_day`, `cumulative_smoking_end`
- **Генетические риски**: все 6 (`genetic_risk_cvd`, `genetic_risk_diabetes`, `genetic_risk_stroke`, `genetic_risk_nafld`, `genetic_risk_colorectal`, `genetic_risk_cirrhosis`)
- **Риски и исходы**: все 6 рисков и исходов (как в файле 4)
- **Дополнительно**: `health_score`, `primary_death_cause`, `estimated_event_age`

---

## Калибровка и настройка

Все коэффициенты в модели намеренно снижены по сравнению с эпидемиологическими данными, чтобы предотвратить доминирование одного фактора и обеспечить сбалансированное обучение ML-моделей. Ключевой настраиваемый параметр:

- **Порог диабета** (`diabetes_threshold`) — регулирует распространённость диабета. Значение 18.7 даёт ~15% заболеваемости за 20 лет.

Остальные коэффициенты (интерцепт SCORE2, влияние курения, алкоголя, натрия) подробно описаны в [научной справке](SCIENTIFIC_REFERENCE_RU.md).

---

## Документация

Подробное описание всех формул, коэффициентов, допущений и методов калибровки приведено в файле [`SCIENTIFIC_REFERENCE.md`](SCIENTIFIC_REFERENCE_RU.md). Рекомендуется ознакомиться перед использованием данных для исследований.

---

## Лицензия

Данные и код распространяются под лицензией [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). Вы можете свободно использовать, изменять и распространять материалы без ограничений.
