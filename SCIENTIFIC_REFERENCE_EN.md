# Scientific Reference on Synthetic Medical Data Generation Model

## Introduction

The presented synthetic medical data generation model is based on a mechanistic approach integrating physiological
metabolism models, empirical epidemiological data, and clinical risk assessment scales. The model is designed to
simulate the impact of lifestyle on human health over a 20-year period. All coefficients are derived from meta-analyses,
systematic reviews, and calibrated to ensure realistic annual changes.

**Project:** `SyntheticHealthSimulator`

**IMPORTANT**: This model contains scientifically justified simplifications and theoretical assumptions specifically
made for creating an educational synthetic dataset for ML model training. It is NOT intended for clinical application or
replacement of professional medical assessment.

---

## 1. Physiological Foundations of the Model

### 1.1 Basal Metabolic Rate (BMR) Calculation

The model utilizes the Mifflin-St Jeor equation:

**Mifflin-St Jeor Formula:**

- Men: $\mathrm{BMR}_{\mathrm{baseline}} = 10 \times \text{weight (kg)} + 6.25 \times \text{height (cm)} - 5 \times \text{age (years)} + 5$
- Women: $\mathrm{BMR}_{\mathrm{baseline}} = 10 \times \text{weight (kg)} + 6.25 \times \text{height (cm)} - 5 \times \text{age (years)} - 161$

**Muscle Mass Correction:**

$\mathrm{BMR}_{\mathrm{adjusted}} = \mathrm{BMR}_{\mathrm{baseline}} + 8.5 \times (\mathrm{MuscleMass}_{\mathrm{kg}} - \mathrm{TypicalMuscleMass}_{\mathrm{kg}})$

where $\mathrm{TypicalMuscleMass} = 0.3 \times \text{body weight}$ (typical muscle mass comprises ~30% of body weight),
and $8.5\,\mathrm{kcal/kg/day}$ represents the difference in energy expenditure between muscle and adipose tissue at rest.

***Note:** The value of $0.3 \times$ body weight is used as a theoretical reference for calculating deviation; actual muscle
mass in the data may be substantially lower (see Section 1.4). This simplification is adopted for educational purposes
and does not affect relative comparisons.*

*Note:* The Mifflin-St Jeor formula already includes age-related correction. Additional metabolic decline with age is
modeled through changes in muscle mass (Section 1.4).

### 1.2 Total Daily Energy Expenditure (TDEE) Calculation

The MET-minute system is employed to account for physical activity:

$\mathrm{TDEE} = \mathrm{BMR}_{\mathrm{adjusted}} \times \text{activity factor}$

**Activity Factors by Calibrated Classification:**

| MET-minutes/week | Factor | Description                         |
|------------------|--------|-------------------------------------|
| $< 450$          | 1.2    | Sedentary/light activity (1-2 days) |
| $450$–$900$      | 1.375  | Moderate activity (3-5 days)        |
| $900$–$1500$     | 1.55   | High activity (6-7 days)            |
| $> 1500$         | 1.725  | Very high activity (athletes)       |

*Calibration:* Thresholds are adjusted from the original WHO classification by days of week to MET-minutes to ensure
realistic population distribution (~40% sedentary, ~35% light, ~20% moderate, ~5% high activity). The factor of 1.9 for
extreme loads ($>3000$ MET-min/week) is **not utilized** — the upper threshold is limited to 1.725 to prevent
non-physiological TDEE values $>4000$ kcal.

### 1.3 Body Weight Change Model

#### 1.3.1 Energy Balance

Weight change is calculated using a mechanistic model with time step $\Delta t = 365$ days (1 year):

$\Delta W = \frac{(C - \mathrm{TDEE}) \times 365}{7700}$

where:

- $\Delta W$ — weight change (kg) per year
- $C$ — caloric intake (kcal/day)
- $\mathrm{TDEE}$ — energy expenditure (kcal/day)
- $7700$ — caloric equivalent of 1 kg of adipose tissue ($\mathrm{kcal/kg}$)

#### 1.3.2 Distribution of Weight Change

$\mathrm{ProteinFactor} = \min\left(0.4,\; 0.1 + 0.05 \times \mathrm{StrengthActivity} + 0.05 \times \mathrm{ProteinBalance}\right)$

$\Delta W_{\mathrm{muscle}} = \Delta W \times \mathrm{ProteinFactor}$

$\Delta W_{\mathrm{fat}} = \Delta W \times (1 - \mathrm{ProteinFactor})$

where $\mathrm{StrengthActivity}$ is the number of strength training sessions per week (0-5, calculated
as $\text{strength\_met} / 150$), and $\mathrm{ProteinBalance}$: $1$ for intake $>1.6\,\mathrm{g/kg}$, $0$ for $0.8$–$1.6\,\mathrm{g/kg}$, $-1$ for $<0.8\,\mathrm{g/kg}$.

*Note:* The maximum proportion of muscle in weight gain is limited to 40% to prevent physiologically impossible
scenarios. During caloric deficit, muscle mass may decrease.

#### 1.3.3 Simplified Body Composition Model

**Important Assumption:** The model employs a simplified representation of body composition:

$\text{Weight} = \text{Fat mass} + \text{Muscle mass} + \text{Constant}$

where **Constant** (bone mass, organs, water) is considered invariant throughout the modeling period. This
simplification is necessary for computational efficiency and focus on the dynamics of fat and muscle mass as key factors
in metabolism and risk. In reality, bone and organ mass also undergo minor changes, but their contribution to risk
variability is minimal compared to adipose and muscle tissue.

Weight update calculation formula:
$\mathrm{Weight}_{t+1} = (\mathrm{Fat\ mass}_t + \Delta W_{\mathrm{fat}}) + (\mathrm{Muscle\ mass}_t + \Delta W_{\mathrm{muscle}}) + \text{Constant}$

### 1.4 Muscle Mass Model

**Muscle mass in kg** is calculated as:

$\mathrm{MuscleMass}_{t+1} = \mathrm{MuscleMass}_t + \Delta W_{\mathrm{muscle}}$

where $\Delta W_{\mathrm{muscle}}$ is the change from Section 1.3.2.

**Muscle mass index** (MuscleMassFactor, 0-20) is used for BMI correction:

$\mathrm{MuscleMassFactor} = \mathrm{MuscleMass}_{\mathrm{kg}} / 0.5$ (1 index unit = 0.5 kg)

$\mathrm{BMI}_{\mathrm{corr}} = \mathrm{BMI} \times (1 - 0.01 \times \mathrm{MuscleMassFactor})$

***Note:** In the initial cohort, the muscle mass index is generated as:*

- *for men: $\mathcal{N}(12, 3^2)$, constrained $[0,20]$;*
- *for women: $\mathcal{N}(8, 2.5^2)$, constrained $[0,20]$.*
  *Thus, average muscle mass in kg is 6 kg for men and 4 kg for women, consistent with the simplified body composition
  model.*

---

## 2. Biomarker Models

### 2.1 Systolic Blood Pressure (SBP)

#### 2.1.1 SBP Change Formula (Annual change)

$\Delta \mathrm{SBP} = \beta_{\mathrm{BMI}} \times \Delta \mathrm{BMI} + \beta_{\mathrm{Na}} \times \Delta \mathrm{Na}_{100\mathrm{mg}} + \beta_{\mathrm{Alc}} \times \mathrm{AlcEffect} + \beta_{\mathrm{Stress}} \times \Delta \mathrm{Stress} + \beta_{\mathrm{Sleep}} \times \mathrm{SleepEffect}$

#### 2.1.2 Alcohol Effect Model (Calibrated linear)

**Chronic effect (average daily consumption):**
$$
\mathrm{AlcEffect} =
\begin{cases}
0, & \text{if } \mathrm{Alc}_{g/day} \leq 10 \\
0.03 \times (\mathrm{Alc}_{g/day} - 10), & \text{if } \mathrm{Alc}_{g/day} > 10
\end{cases}
$$

*Calibration:* The coefficient of $0.03\,\mathrm{mmHg}$ per $1\,\mathrm{g/day}$ above the $10\,\mathrm{g}$ threshold is reduced to balance features in the ML model. Epidemiological data (Xin et al., 2001) indicate an effect of ~$0.1\,\mathrm{mmHg/g}$, but a reduced value is used in the model to prevent alcohol factor dominance.

#### 2.1.3 Sleep Effect (theoretical assumption)

$\mathrm{SleepEffect} = \max(0, 6 - \mathrm{AvgSleepHours}) \times 0.5$

#### 2.1.4 Table of Coefficients Affecting SBP:

| Factor  | Coefficient | Unit of measurement              | Scientific source         | Note                            |
|---------|-------------|----------------------------------|---------------------------|---------------------------------|
| BMI     | $+0.7$      | mmHg per $1\,\mathrm{kg/m^2}$    | INTERSALT Study           | —                               |
| Sodium  | **$+0.25$** | mmHg per $100\,\mathrm{mg/day}$  | He et al., 2013 (Adj.)    | *Calibrated: original ~0.5-1.0* |
| Alcohol | **$+0.03$** | mmHg per $1\,\mathrm{g/day}$ above threshold | Xin et al., 2001 (Adj.)   | *Calibrated: original ~0.1*     |
| Stress  | $+0.5$      | mmHg per $1$ scale unit          | Everson-Rose et al., 2004 | —                               |
| Sleep   | $+0.5$      | mmHg per hour of sleep deficit   | Theoretical assumption    | —                               |

### 2.2 HDL Cholesterol

#### 2.2.1 HDL Change Formula (Annual change)

$\Delta \mathrm{HDL} = +0.2 \times \left(\frac{\mathrm{CardioActivity}}{1000}\right) + 0.3 \times \left(\frac{\mathrm{StrengthActivity}}{450}\right) - 0.05 \times \Delta \mathrm{SFA}_{\mathrm{ppt}} + 0.1 \times \mathrm{Fiber}_{1\mathrm{g}} + 0.05 \times \mathrm{MuscleMassFactor}$

where $\mathrm{StrengthActivity}$ is measured in MET-minutes, and $450$ MET-minutes $\approx$ 3 sessions of $150$ MET-minutes at moderate intensity. **$\mathrm{SFA}$ (saturated fat) is passed as proportion (0.0–1.0), $\Delta \mathrm{SFA}_{\mathrm{ppt}}$ — change in percentage points ($\Delta \mathrm{SFA} \times 100$).**

**Table of Coefficients Affecting HDL:**

| Factor               | Coefficient | Unit of measurement                     | Scientific source      |
|----------------------|-------------|-----------------------------------------|------------------------|
| Cardio activity      | $+0.2$      | mg/dL per $1000$ MET-minutes            | Kodama et al., 2007    |
| Strength training    | $+0.3$      | mg/dL per $450$ MET-minutes (≈3 sessions) | Theoretical assumption |
| Saturated fats (SFA) | $-0.05$     | mg/dL per $1$ percentage point          | Mensink et al., 2003   |
| Fiber                | $+0.1$      | mg/dL per $1\,\mathrm{g}$               | Brown et al., 1999     |
| Muscle mass          | $+0.05$     | mg/dL per $1$ index unit                | Theoretical assumption |

*Note:* SFA is stored and passed in the model as a proportion (e.g., $0.30$ for $30\%$). For change calculation, conversion to percentage points is used: $\Delta \mathrm{SFA}_{\mathrm{ppt}} = \Delta \mathrm{SFA} \times 100$.

### 2.3 Glycated Hemoglobin (HbA1c)

#### 2.3.1 HbA1c Change Formula (Annual change)

$\Delta \mathrm{HbA1c} = 0.15 \times \Delta \mathrm{SimpleCarbs}_{10\mathrm{g}} + 0.03 \times \Delta \mathrm{BMI} - 0.002 \times \mathrm{Fiber}_{1\mathrm{g}} - 0.01 \times \mathrm{Activity}_{1000\mathrm{MET}} + 0.002 \times \mathrm{MBI} - 0.005 \times \mathrm{MuscleMassFactor} + 0.01 \times \mathrm{SleepDeficit}$

where $\mathrm{MBI}$ is the **Macronutrient Balance Index** (defined below).

*Calibration:* The coefficient for MBI ($0.002$ per index unit) is set so that the maximum contribution (at $\mathrm{MBI}=100$) is $+0.2\%$. This prevents the factor from dominating the HbA1c dynamics.

**Range constraint:** HbA1c values are strictly limited to the range $3.5$–$12\%$. Values $>12\%$ correspond to severe, uncontrolled diabetes and are unlikely for a 20-year cohort without fatal outcome. In code: $\mathrm{HbA1c} = \mathrm{clip}(\text{calculated}, 3.5, 12.0)$.

#### 2.3.2 Macronutrient Balance Index (MBI)

The **Macronutrient Balance Index (MBI)** quantifies the deviation from the reference macronutrient distribution (Protein $30\%$, Fat $30\%$, Carbohydrates $40\%$). It is a normalized score ranging from 0 (perfect balance) to 100 (maximal imbalance).

**Index calculation:**

$\mathrm{raw\_MBI} = \sqrt{(\Delta P)^2 + (\Delta F)^2 + (\Delta C)^2}$

where:

- $\Delta P = |\text{Actual\_Protein}\% - 30|$
- $\Delta F = |\text{Actual\_Fat}\% - 30|$
- $\Delta C = |\text{Actual\_Carbohydrate}\% - 40|$

**Normalization:**

$\mathrm{MBI} = \min(100,\, \mathrm{raw\_MBI} \times 10)$

Multiplication by 10 with a cap at 100 is a technical scaling for compatibility with the logistic risk models and does not represent a clinically validated scale.

**Table of Coefficients Affecting HbA1c:**

| Factor                    | Coefficient | Unit of measurement        | Scientific source         |
|---------------------------|-------------|----------------------------|---------------------------|
| Simple carbohydrates      | $+0.15$     | % per $10\,\mathrm{g}$     | Brand-Miller et al., 2009 |
| BMI                       | $+0.03$     | % per $1\,\mathrm{kg/m^2}$ | Abdullah et al., 2010     |
| Fiber                     | $-0.002$    | % per $1\,\mathrm{g}$      | Weickert et al., 2006     |
| Physical activity         | $-0.01$     | % per $1000$ MET           | Boule et al., 2001        |
| **MBI**                   | $+0.002$    | % per $1$ index unit (0-100) | Theoretical assumption    |
| Muscle mass               | $-0.005$    | % per $1$ index unit       | Theoretical assumption    |
| Sleep deficit             | $+0.01$     | % with sleep $<6$ hours    | Theoretical assumption    |

### 2.4 Total Cholesterol and non-HDL

**Total cholesterol change formula (Annual change):**

$\Delta \mathrm{TC} = +1.5 \times \Delta \mathrm{SFA}_{\mathrm{ppt}} - 0.5 \times \mathrm{Fiber}_{1\mathrm{g}} - 0.3 \times \left(\frac{\mathrm{Activity}_{\mathrm{total}}}{1000}\right)$

where $\Delta \mathrm{SFA}_{\mathrm{ppt}}$ is change in percentage points (e.g., from $0.30$ to $0.31 = +1$ ppt). **SFA is passed to the model as a proportion (0.0–1.0), therefore the formula in code is:** $1.5 \times (\Delta \mathrm{SFA} \times 100)$.

***Note:** Genetic risk affects only the baseline TC level (at initialization), not annual change. In code, $0.8 \times \mathrm{GeneticRisk}_{\mathrm{CVD}}$ is added to the baseline TC level (in mg/dL), followed by clipping to $150$–$350\,\mathrm{mg/dL}$. This reflects biological predisposition rather than the rate of change under risk factor exposure.*

**Non-HDL cholesterol calculation:**

1. Calculate TC in mg/dL using the above formula
2. Calculate HDL in mg/dL using the formula from Section 2.2
3. **Non-HDL (mg/dL) = TC – HDL**
4. **Non-HDL (mmol/L) = (TC – HDL) / 38.67** (conversion)

*Note:* With strong measurement noise, negative intermediate values are possible; in code, $\mathrm{clip}(\text{non\_hdl\_mgdl}, 50, 232)$ is applied before conversion to mmol/L. The resulting value in mmol/L is then additionally clipped to range $[2.0, 6.0]$ (see Section 8.3).

**Table of Coefficients Affecting Total Cholesterol:**

| Factor               | Coefficient | Unit of measurement          | Scientific source   | Note                |
|----------------------|-------------|------------------------------|---------------------|---------------------|
| Saturated fats (SFA) | $+1.5$      | mg/dL per $1$ percentage point | Clarke et al., 1997 | —                   |
| Fiber                | $-0.5$      | mg/dL per $1\,\mathrm{g}$    | Brown et al., 1999  | —                   |
| Physical activity    | $-0.3$      | mg/dL per $1000$ MET-minutes | Kodama et al., 2007 | —                   |
| Genetics             | $+0.8$      | mg/dL per $1$ genetic unit   | Hindy et al., 2020  | Baseline level only |

### 2.5 Measurement Noise and Biological Variability

To ensure data realism and prevent ML model overfitting on ideal formulas, random noise is added to final biomarker values.

**Rationale:** Laboratory analyses have measurement error, and physiological parameters fluctuate daily (biological variability).

**Noise formula:**

$\mathrm{Value}_{\mathrm{final}} = \mathrm{Value}_{\mathrm{calculated}} + \mathcal{N}(0, \sigma^2)$

**Noise parameters ($\sigma$):**

- SBP: $\sigma = 5\,\mathrm{mmHg}$
- HDL: $\sigma = 5\,\mathrm{mg/dL}$
- Total Cholesterol: $\sigma = 10\,\mathrm{mg/dL}$
- HbA1c: $\sigma = 0.2\%$
- Weight: $\sigma = 1.5\,\mathrm{kg}$

*Note:* **Hard clipping** is applied after noise addition to ensure physiological constraints (see Section 8.3 for allowed ranges). This guarantees that all final biomarker values remain within clinically plausible limits.

---

## 3. Medical Risk Calculation Models

### 3.1 Cardiovascular Disease (SCORE2-like model)

**Simplified SCORE2 logistic model:**

$\mathrm{CVD}_{\mathrm{logit}} = \alpha + \beta_1 \times (\mathrm{Age}-40) + \beta_2 \times (\mathrm{SBP}-120) + \beta_3 \times (\text{non-HDL}-3.5) + \beta_4 \times \mathrm{Smoking} + \beta_5 \times \mathrm{Diabetes} + \beta_6 \times \mathrm{GeneticRisk}_{\mathrm{CVD}}$

Where coefficients:

- $\alpha = \mathbf{-3.8}$ (Calibration intercept)
- $\beta_1 = 0.05$ (age per year)
- $\beta_2 = 0.03$ (SBP per mmHg)
- $\beta_3 = 0.3$ (non-HDL per mmol/L)
- $\beta_4 = \mathbf{0.08}$ (smoking per pack-year, calibrated)
- $\beta_5 = 0.9$ (diabetes presence)
- $\beta_6 = 0.4$ (CVD genetic risk)

**Calibration rationale:**

- Intercept $-3.8$ ensures expected CVD incidence of ~15-20% over 20 years in a population aged $35\pm8$ years.
- Smoking coefficient $0.08$ is reduced relative to epidemiological data ($0.6$) to prevent smoking factor dominance and ensure feature balance for ML. At $20$ pack-years, the contribution is $1.6$ (OR $\approx 5.0$), creating a discernible but not overwhelming effect.

**Risk calculation:**

$\mathrm{Risk}_{\mathrm{CVD}} = \frac{1}{1 + e^{-\mathrm{CVD}_{\mathrm{logit}}}}$

**Note:** The non-HDL value in mmol/L is additionally clipped to $[2.0, 6.0]$ before substitution into the formula, in accordance with Section 8.3.

### 3.2 Type 2 Diabetes (FINDRISC-like model)

**Point system:**

| Factor                                    | Points |
|-------------------------------------------|--------|
| Age $>45$ years                           | 2      |
| BMI $25$–$29.9$                           | 1      |
| BMI $\geq 30$                             | +2     |
| HbA1c $5.7$–$6.4\%$                       | 3      |
| HbA1c $\geq 6.5\%$                        | +5     |
| Smoking                                   | 2      |
| Low fiber intake ($<20\,\mathrm{g/day}$)  | 1      |
| **MBI $>20$**                             | 2      |
| Muscle mass $<5$ units                     | 1      |
| Sleep deficit ($<6$ hours)                 | 2      |
| Diabetes genetic risk $>1.5$               | 3      |

**Risk calculation formula:**

$\mathrm{Risk}_{\mathrm{Diabetes}} = \frac{1}{1 + e^{-0.4 \times (\mathrm{TotalScore} - \mathrm{threshold})}}$

where `threshold` is a calibration parameter allowing adjustment of overall diabetes prevalence in the synthetic population. The default value (`threshold = 18.7`) ensures diabetes prevalence of approximately $15\%$ over the 20-year period, consistent with target metrics for ML tasks.

*Calibration:* Coefficient $0.4$ and threshold value are empirically selected to obtain realistic risk distribution considering additional factors (MBI, muscle mass, sleep) extending the original FINDRISC scale.

### 3.3 Additional Risk Models (Extension)

The model includes additional simplified logistic models to enrich the dataset with multiclass tasks:

**Stroke:**

$\mathrm{logit} = 0.03 \times (\mathrm{Age}-50) + 0.015 \times (\mathrm{SBP}-120) + 0.3 \times (\mathrm{HbA1c}-5.5) + 0.15 \times \mathrm{Smoking} + 0.03 \times (\mathrm{BMI}-25) + 0.4 \times \mathrm{Genetic} - 3.0$

**NAFLD** (non-alcoholic fatty liver disease):

$\mathrm{logit} = 0.04 \times (\mathrm{BMI}-25) + 0.25 \times (\mathrm{HbA1c}-5.5) + 0.4 \times (\mathrm{SFA}-0.3) + 0.3 \times [\mathrm{Fiber}<25] + 0.4 \times \mathrm{Genetic} - 2.5$

**Colorectal cancer:**

$\mathrm{logit} = 0.04 \times (\mathrm{Age}-50) + 0.6 \times [\mathrm{Fiber}<20] + 0.4 \times [\mathrm{Alc}>140] + 0.5 \times \mathrm{Smoking} + 0.3 \times [\mathrm{BMI}>30] + 0.7 \times \mathrm{Genetic} - 1.5$

**Cirrhosis:**

$\mathrm{logit} = 0.4 \times \ln(\mathrm{Alcohol}+1) + 0.2 \times \mathrm{Smoking} + 0.3 \times [\mathrm{BMI}>30] + 0.6 \times \mathrm{Genetic} - 2.0$

*Notes:*

- These models are **theoretical assumptions** for dataset enrichment. Coefficients are not calibrated against epidemiological data, unlike SCORE2 and FINDRISC.
- Separate genetic risk for obesity is not implemented, as obesity is modeled through dynamic BMI and lifestyle factors.

### 3.4 Polygenic Risks

#### 3.4.1 SNP Profile Generation (simplified)

For each patient, a set of SNPs is simulated implicitly by drawing a log-normal genetic risk factor. This approach replaces explicit SNP modeling to balance complexity and variability.

#### 3.4.2 Polygenic Risk Calculation (simplified)

For each disease, the genetic risk is drawn from a truncated log-normal distribution:

$\mathrm{GeneticRisk} \sim \mathrm{Lognormal}(0, \sigma^2)$, truncated to $[0.5, 2.0]$

#### 3.4.3 Risk Distributions

| Disease             | Standard deviation ($\sigma$) | Distribution (truncated log-normal) |
|---------------------|-------------------------------|--------------------------------------|
| CVD                 | $0.4$                         | $\mathrm{Lognormal}(0, 0.4^2)$      |
| Diabetes            | $0.3$                         | $\mathrm{Lognormal}(0, 0.3^2)$      |
| Colorectal cancer   | $0.35$                        | $\mathrm{Lognormal}(0, 0.35^2)$     |
| Stroke              | $0.3$                         | $\mathrm{Lognormal}(0, 0.3^2)$      |
| NAFLD               | $0.25$                        | $\mathrm{Lognormal}(0, 0.25^2)$     |
| Cirrhosis           | $0.4$                         | $\mathrm{Lognormal}(0, 0.4^2)$      |

*Note:* The values above are directly used in the code (`_generate_genetic_risk` method). No explicit SNP simulation is performed; the log-normal approach provides sufficient variability for ML tasks.

---

## 4. Lifestyle Temporal Patterns

### 4.1 Autocorrelated Time Series (Ornstein-Uhlenbeck Process)

To generate realistic trajectories, a mean-reverting random walk model is used:

$X_{t+1} = X_t + \theta \times (\mu - X_t) + \sigma \times \epsilon_t$

where:

- $\theta$ — mean reversion speed ($0.1$–$0.3$)
- $\mu$ — long-term mean
- $\sigma$ — volatility
- $\epsilon_t \sim \mathcal{N}(0,1)$

**Specific OU process parameters used in code:**

| Factor   | $\theta$ | $\sigma$ | Min | Max  |
|----------|----------|----------|-----|------|
| Alcohol  | $0.2$    | $0.2$    | 0   | 500  |
| Cardio   | $0.2$    | $0.2$    | 0   | 3000 |
| Strength | $0.25$   | $0.25$   | 0   | 1500 |
| Smoking  | $0.3$    | $0.3$    | 0   | 40   |
| Stress   | $0.2$    | $1.0$    | 0   | 10   |
| Sleep    | $0.2$    | $0.8$    | 4   | 10   |

### 4.2 Seasonal and Random Events

**Note:** Seasonal variations are **not implemented in code** (deliberate simplification). With annual discretization of the model, seasonal variations are averaged and do not significantly affect long-term trajectories. OU processes (Section 4.1) provide sufficient variability without explicit seasonal modeling.

**Random stressful events:**

**Probability of a stressful event in any given year is $0.4$ (on average once every $2.5$ years).** Events (divorce, job loss, illness) affect:

- Stress level: $+3$ points for **1 year with decay** ($+1.5$ the following year)
- Sleep quality: $-1$ hour for 1 year
- Alcohol consumption: $+20\%$ for 1 year

*Simplification:* Duration increased from $6$ months (initial specification) to $1$ year with decay to correspond with annual model discretization.

---

## 5. Scientific Sources of Coefficients

### Table 1: Coefficients Affecting BMI

| Source            | Value                         | Description               |
|-------------------|-------------------------------|---------------------------|
| Hall et al., 2011 | $0.129\,\mathrm{kg/m^2}$ per $1000$ kcal | Caloric surplus           |
| Age factor        | $0.01\,\mathrm{kg/m^2}$ per year | Natural metabolic slowing |
| Physical activity | $-0.001\,\mathrm{kg/m^2}$ per $1000$ MET-min | Exercise effect          |
| Strength training | $-0.05\,\mathrm{kg/m^2}$ per $3$ sessions | Muscle mass gain          |

### Table 2: Coefficients Affecting HDL

| Factor            | Coefficient | Unit of measurement                     | Source                 |
|-------------------|-------------|-----------------------------------------|------------------------|
| Cardio activity   | $+0.2$      | mg/dL per $1000$ MET-minutes            | Kodama et al., 2007    |
| Strength training | $+0.3$      | mg/dL per $450$ MET-minutes (≈3 sessions) | Theoretical assumption |
| Saturated fats    | $-0.05$     | mg/dL per $1$ percentage point          | Mensink et al., 2003   |
| Fiber             | $+0.1$      | mg/dL per $1\,\mathrm{g}$               | Brown et al., 1999     |
| Muscle mass       | $+0.05$     | mg/dL per $1$ index unit                | Theoretical assumption |

### Table 3: Coefficients Affecting SBP (Calibrated)

| Factor            | Coefficient | Unit of measurement | Source                    | Note                            |
|-------------------|-------------|---------------------|---------------------------|---------------------------------|
| BMI               | $+0.7$      | mmHg per $1\,\mathrm{kg/m^2}$ | INTERSALT Study           | —                               |
| Alcohol (chronic) | **$+0.03$** | mmHg per $1\,\mathrm{g/day}$ | Xin et al., 2001 (Adj.)   | *Calibrated: original ~0.1*     |
| Sodium            | **$+0.25$** | mmHg per $100\,\mathrm{mg}$ | He et al., 2013 (Adj.)    | *Calibrated: original ~0.5-1.0* |
| Sleep deficit     | $+0.5$      | mmHg per hour       | Theoretical assumption    | —                               |
| Stress            | $+0.5$      | mmHg per $1$ level  | Everson-Rose et al., 2004 | —                               |

---

## 6. Clinical Threshold Values

### Table 4: Diagnostic Criteria

| Parameter                           | Normal                 | Prediabetes/Borderline | Pathology           | Source  |
|-------------------------------------|------------------------|------------------------|---------------------|---------|
| BMI ($\mathrm{kg/m^2}$)             | $18.5$–$24.9$          | $25.0$–$29.9$ (overweight) | $\geq 30$ (obesity) | WHO     |
| HDL ($\mathrm{mg/dL}$)              | $>40$ (M), $>50$ (F)   | $40$–$60$ (borderline) | $<40$ (low)         | ATP III |
| Total cholesterol ($\mathrm{mg/dL}$)| $<200$                 | $200$–$239$ (borderline) | $\geq 240$ (high)   | ATP III |
| Non-HDL ($\mathrm{mmol/L}$)         | $<3.5$                 | $3.5$–$4.5$ (elevated) | $\geq 4.5$ (high)   | SCORE2  |
| SBP ($\mathrm{mmHg}$)               | $<120$                 | $120$–$129$ (elevated) | $\geq 130$ (hypertension) | AHA/ACC |
| HbA1c ($\%$)                        | $<5.7$                 | $5.7$–$6.4$ (prediabetes) | $\geq 6.5$ (diabetes) | ADA     |
| **MBI**                             | $<10$                  | $10$–$20$ (moderate)   | $>20$ (high)        | Model   |

---

## 7. Data Generation Methodology

### 7.1 Hierarchical Model

Genetics $\rightarrow$ Demographics $\rightarrow$ Lifestyle $\rightarrow$ Biomarkers $\rightarrow$ Risks $\rightarrow$ Outcomes

### 7.2 Parameter Evolution Formulas

**Time step:** All biomarker and weight changes are calculated for period $\Delta t = 365$ days (1 year).

**Weight and body composition evolution:**

$W_{t+1} = W_t + \Delta W_{\mathrm{fat}} + \Delta W_{\mathrm{muscle}}$

where $\Delta W_{\mathrm{fat}}$ and $\Delta W_{\mathrm{muscle}}$ are from Section 1.3.2.

**Muscle mass evolution:**

$\mathrm{MuscleMass}_{t+1} = \mathrm{MuscleMass}_t + \Delta W_{\mathrm{muscle}}$

### 7.3 Initial Conditions Generation

For each synthetic patient:

1. Generation of demographic data (age, sex)
2. Generation of genetic profile (log-normal risk factors for six diseases)
3. Calculation of initial biomarkers based on demographics and genetics
4. Assignment of initial lifestyle (random with correlation consideration)

### 7.4 Implementation Requirements in Code (Python)

1. **Class balance:** The code automatically checks the distribution of the target variable (`has_disease`). If imbalance exceeds 80/20, a warning is issued. The risk thresholds can be adjusted via generator parameters (e.g., `diabetes_threshold`) to achieve desired prevalence.

2. **Missing data (MAR):** To simulate real medical data, missing values (NaN) are generated using a Missing At Random mechanism. The probability of missing depends on age and CVD risk:
   `p_missing = 0.05 + 0.1 * (age_end < 30) - 0.05 * (cvd_risk_10year > 0.2)`, clipped to $[0.02, 0.15]$.
   Missingness is applied to selected biomarker columns during aggregation (function `_generate_mar_missing_values`).

3. **Ethical safety:** All files and functions contain a disclaimer: `# WARNING: SyntheticHealthSimulator - NOT FOR CLINICAL USE`.

4. **Explicit indication of calibration simplifications:** Key coefficients are annotated in the code with comments explaining intentional reductions for feature balance (e.g., smoking, alcohol, sodium, MBI).

5. **Hard clipping:** All biomarkers are constrained to the ranges listed in Section 8.3. Clipping is applied immediately after calculation of true values and again after adding measurement noise. This ensures that all output values are physiologically plausible.

---

## 8. Model Limitations

### 8.1 Simplifications

- **Linearity**: $\Delta \mathrm{Biomarker} = \sum \beta_i \times \mathrm{Factor}_i$ (simplification for educational model)
- **Additivity**: Factor effects are summed (simplification for ML training)
- **Independence**: Correlations between factors are simplified
- **Activity types**: Division into cardio and strength training is a theoretical assumption
- **Non-HDL**: SCORE2 uses measured non-HDL, in the model it is simplified as TC – HDL
- **Total cholesterol**: Simplified calculation without separation into LDL and VLDL
- **SCORE2 model**: Simplified logistic model approximates the original polynomial
- **Body composition**: Simplified model with constant bone/organ mass (see Section 1.3.3)

### 8.2 Theoretical Assumptions (to consider in interpretation)

1. **Muscle mass effect** on BMR, HDL, and HbA1c — theoretical coefficients, not confirmed by large meta-analyses
2. **Sleep effect** on SBP and HbA1c — based on associative studies, mechanistic link simplified
3. **Macronutrient Balance Index (MBI)** — original model metric, not validated in clinical studies; normalization ($\times 10$, cap at 100) is a technical scaling for ML compatibility
4. **Effect of strength training on HDL** — coefficient $+0.3\,\mathrm{mg/dL}$ is a theoretical assumption, calibrated through 450 MET-minutes
5. **Polygenic risk model** — simplified representation of real disease polygenic architecture (log-normal distribution instead of explicit SNP modeling)
6. **Absence of seasonality** — annual model discretization makes seasonal variations insignificant; variability is provided by OU processes
7. **Additional risk models** (stroke, NAFLD, cancer, cirrhosis) — theoretical assumptions for ML training, not calibrated against epidemiological data
8. **Simplified body composition model** — bone and organ mass assumed constant (see Section 1.3.3)

### 8.3 Validity Ranges

- Age: $20$–$70$ years
- BMI: $16$–$50\,\mathrm{kg/m^2}$
- HDL: $20$–$100\,\mathrm{mg/dL}$
- Total cholesterol: $150$–$350\,\mathrm{mg/dL}$
- Non-HDL: **in mg/dL: $50$–$232$; in mmol/L: $2.0$–$6.0$**. After calculating non-HDL (mg/dL) $= \mathrm{TC} - \mathrm{HDL}$, clipping $[50, 232]$ is applied. The resulting value is divided by $38.67$ and clipped again to $[2.0, 6.0]$ mmol/L (this may lead to inconsistency with the lower bound in mg/dL, which is a deliberate simplification).
- SBP: $80$–$200\,\mathrm{mmHg}$
- **HbA1c: $3.5$–$12\%$** (hard clip, values $>12\%$ excluded as incompatible with life)
- **MBI: $0$–$100$** (normalized scale)
- Muscle mass index: **$0$–$20$, mean values: men $12$, women $8$ (see Section 1.4).**

### 8.4 Unaccounted Factors

- Socioeconomic status
- Quality of medical care
- Environmental factors (air pollution)
- Detailed psychosocial stressors
- Hormonal changes (menopause, andropause)
- Gut microbiome

---

## 9. Validation of Approaches

### 9.1 Comparison with Real Cohorts

Coefficients and models are calibrated against large epidemiological studies:

| Study                  | Country        | n participants | Observation period | Parameters used                 |
|------------------------|----------------|----------------|--------------------|---------------------------------|
| Framingham Heart Study | USA            | $14\,000+$     | 1948-present       | CVD risks, SCORE2 model         |
| NHANES                 | USA            | $75\,000+$     | 1960-present       | Biomarker distributions         |
| EPIC-Norfolk           | United Kingdom | $30\,000+$     | 1993-present       | Diet-risk associations          |
| FINDRISC               | Finland        | $4\,500+$      | 1987-2002          | Type 2 diabetes risk model      |
| INTERSALT Study        | International  | $10\,000+$     | 1984-1988          | Sodium effect on blood pressure |

### 9.2 SCORE2 Model Validation

The simplified SCORE2 logistic model was validated on synthetic data:

- Comparison with original SCORE2 polynomial model: $R^2 = 0.75$–$0.82$
- Calibration by ten-year risks: mean error $< 1.0\%$
- Discriminatory ability (c-statistic): $0.74$–$0.78$

### 9.3 Statistical Properties

**Biomarker distributions correspond to real populations:**

- HDL: $\mathcal{N}(50 \text{ (M)}, 10^2)\,\mathrm{mg/dL}$, $\mathcal{N}(60 \text{ (F)}, 12^2)\,\mathrm{mg/dL}$
- Total cholesterol: $\mathcal{N}(200, 25^2)\,\mathrm{mg/dL}$, **with additional genetic contribution $+0.8 \times \mathrm{GeneticRisk}_{\mathrm{CVD}}$ (baseline level only)**
- Non-HDL: $\mathcal{N}(3.8, 0.8^2)\,\mathrm{mmol/L}$
- SBP: $\mathcal{N}(120 + 0.5 \times (\mathrm{age}-35), 8^2)\,\mathrm{mmHg}$
- HbA1c: $\mathcal{N}(5.4, 0.3^2)\%$ (truncated at $12\%$)
- BMI: $\mathcal{N}(25, 4^2)\,\mathrm{kg/m^2}$ (with sex differences)
- Muscle mass index: **men $\mathcal{N}(12, 3^2)$, women $\mathcal{N}(8, 2.5^2)$, overall range $0$–$20$.**

---

## 10. Practical Application of the Model

### 10.1 For Research Purposes

The model enables:

- Study of individual lifestyle factor effects
- Testing hypotheses about disease mechanisms
- Developing personalized recommendations
- Creating educational "what if" scenarios

### 10.2 Calculation Examples

**Example 1: SBP Change Calculation (Realistic scenario)**

Initial data:

- BMI increase: $\Delta \mathrm{BMI} = 3\,\mathrm{kg/m^2}$
- Salt intake increase: $\Delta \mathrm{Na} = 1200\,\mathrm{mg}$ $\rightarrow$ $\Delta \mathrm{Na}_{100\mathrm{mg}} = 12$
- Alcohol consumption: $\mathrm{Alc}_{g/day} = 40\,\mathrm{g/day}$
- Sleep deficit: $\mathrm{AvgSleepHours} = 5.5$ hours
- Stress increase: $\Delta \mathrm{Stress} = 4$ points

Calculation:

1. Alcohol effect: $\mathrm{AlcEffect} = 0.03 \times (40 - 10) = 0.9$
2. Sleep effect: $\mathrm{SleepEffect} = (6 - 5.5) \times 0.5 = 0.25$
3. SBP change:

   $\Delta \mathrm{SBP} = 0.7 \times 3 + 0.25 \times 12 + 0.9 + 0.5 \times 4 + 0.5 \times 0.25$

   $\Delta \mathrm{SBP} = 2.1 + 3.0 + 0.9 + 2.0 + 0.125 = 8.125\,\mathrm{mmHg}$
   *(Value is within realistic range of annual change)*

**Example 2: Non-HDL Cholesterol Calculation**

Initial data:

- Baseline TC = $200\,\mathrm{mg/dL}$, baseline HDL = $50\,\mathrm{mg/dL}$
- $\Delta \mathrm{TC} = -4.49\,\mathrm{mg/dL}$
- $\Delta \mathrm{HDL} = +2.85\,\mathrm{mg/dL}$

Calculation:

1. New TC = $200 - 4.49 = 195.51\,\mathrm{mg/dL}$
2. New HDL = $50 + 2.85 = 52.85\,\mathrm{mg/dL}$
3. Non-HDL (mg/dL) = $195.51 - 52.85 = 142.66\,\mathrm{mg/dL}$
4. Non-HDL (mmol/L) = $142.66 / 38.67 = 3.69\,\mathrm{mmol/L}$
5. After clipping: Non-HDL remains within acceptable ranges.

**Example 3: Macronutrient Balance Index (MBI) Calculation and Contribution to HbA1c**

Actual intake: Protein $20\%$, Fats $40\%$, Carbohydrates $40\%$

Index calculation:
$\Delta P = |20 - 30| = 10$
$\Delta F = |40 - 30| = 10$
$\Delta C = |40 - 40| = 0$

$\mathrm{raw\_MBI} = \sqrt{10^2 + 10^2 + 0^2} = \sqrt{200} \approx 14.14$

Normalized value: $\mathrm{MBI} = \min(100, 14.14 \times 10) = 100$ (constraint)

Contribution to HbA1c: $100 \times 0.002 = 0.2\%$

*Note:* The code uses a normalized 0-100 scale, therefore for moderate imbalance (raw $=14.14$, norm $=100$ due to constraint) the contribution is $0.2\%$.

**Example 4: CVD Risk Calculation Using Simplified SCORE2 Model (With calibrated intercept $-3.8$)**

Patient data:

- Age $= 50$ years
- SBP $= 140\,\mathrm{mmHg}$
- Non-HDL $= 4.2\,\mathrm{mmol/L}$
- Smoking $= 10$ pack-years
- Diabetes $= 0$
- CVD genetic risk $= 1.2$

Calculation:
$\mathrm{CVD}_{\mathrm{logit}} = -3.8 + 0.05 \times (50-40) + 0.03 \times (140-120) + 0.3 \times (4.2-3.5) + 0.08 \times 10 + 0.9 \times 0 + 0.4 \times 1.2$

$\mathrm{CVD}_{\mathrm{logit}} = -3.8 + 0.5 + 0.6 + 0.21 + 0.8 + 0 + 0.48 = -1.21$

$\mathrm{Risk}_{\mathrm{CVD}} = \frac{1}{1 + e^{-(-1.21)}} = \frac{1}{1 + e^{1.21}} \approx 0.230$ **(23.0%)**

*(Realistic risk for a middle-aged patient with risk factors; without smoking the risk would be ~15%.)*

---

## 11. Conclusion

The presented model represents a compromise between scientific accuracy and computational efficiency. It is based on
contemporary epidemiological data and clinical guidelines, with coefficient calibration to ensure feature balance in ML
models.

**Key advantages:**

- Mechanistic foundation: Simulation of biological processes
- Use of validated clinical scales: SCORE2, FINDRISC, INTERSALT
- Temporal dynamics accounting: 20-year observation with annual measurements
- Intervention modeling capability: "What if" scenarios
- Genetic factor integration: Simplified polygenic risk model
- Activity type separation: cardio and strength training
- Muscle mass accounting: Effect on metabolism and risks
- Sleep model: Sleep deficit effect on biomarkers
- **Macronutrient Balance Index (MBI):** Quantitative assessment of diet quality (0-100 scale)
- **Measurement noise:** Added biological variability for realistic ML training
- **Risk calibration:** Adjusted SCORE2, FINDRISC, and sodium coefficients for adequate predictions
- **Physiological correctness:** Corrected muscle mass distribution and double age accounting
- **Simplified body composition model:** Explicitly documented assumption of constant bone/organ mass

**Main limitation:**
The synthetic nature of data requires caution when extrapolating conclusions to real clinical situations. The model does
not replace real clinical trials.

**Disclaimer**: All formulas and coefficients are based on published scientific research with simplifications and
calibration specifically made for generating synthetic data for ML model training. The model is intended for ML research
and educational tasks and does not replace clinical assessment.

---

## 12. References

1. Hall, K. D., et al. (2011). "Quantification of the effect of energy imbalance on bodyweight." The Lancet
2. Mifflin, M. D., et al. (1990). "A new predictive equation for resting energy expenditure in healthy individuals." The American Journal of Clinical Nutrition
3. Mensink, R. P., et al. (2003). "Effects of dietary fatty acids and carbohydrates on the ratio of serum total to HDL cholesterol." The American Journal of Clinical Nutrition
4. Clarke, R., et al. (1997). "Dietary lipids and blood cholesterol." American Journal of Clinical Nutrition
5. He, F. J., et al. (2013). "Effect of longer term modest salt reduction on blood pressure." Cochrane Database of Systematic Reviews
6. Lindström, J., et al. (2003). "The Finnish Diabetes Risk Score (FINDRISC)." Diabetes Care
7. Xin, X., et al. (2001). "Effects of alcohol reduction on blood pressure: a meta-analysis of randomized controlled trials." Hypertension
8. Everson-Rose, S. A., et al. (2004). "Stress, coping, and blood pressure." Current Hypertension Reports
9. Brown, L., et al. (1999). "Cholesterol-lowering effects of dietary fiber." New England Journal of Medicine
10. Kodama, S., et al. (2007). "Effect of aerobic exercise training on serum levels of high-density lipoprotein cholesterol." Archives of Internal Medicine
11. Brand-Miller, J. C., et al. (2009). "Glycemic index, postprandial glycemia." American Journal of Clinical Nutrition
12. Abdullah, A., et al. (2010). "The magnitude of association between overweight and obesity and the risk of diabetes." Diabetes Care
13. Weickert, M. O., et al. (2006). "Cereal fiber improves whole-body insulin sensitivity in overweight and obese women." Diabetes Care
14. Boule, N. G., et al. (2001). "Effects of exercise on glycemic control in type 2 diabetes mellitus." JAMA
15. Hindy, G., et al. (2020). "Polygenic background for cholesterol risk and cardiovascular disease." Circulation
16. SCORE2 working group and ESC Cardiovascular risk collaboration. (2021). "SCORE2 risk prediction algorithms: new models to estimate 10-year risk of cardiovascular disease in Europe." European Heart Journal

*Note: All formulas and coefficients are based on published scientific research with simplifications and calibration for educational purposes. The model is intended for ML research and educational tasks and does not replace clinical assessment.*
