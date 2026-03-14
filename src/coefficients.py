"""
Scientific coefficients for SyntheticHealthSimulator.
All coefficients are documented with their physiological meaning and units.
Based on meta-analyses and calibrated for ML feature balance.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiomarkerCoefficients:
    """Base class for biomarker coefficients with change limits."""

    CHANGE_MIN: float
    CHANGE_MAX: float


@dataclass(frozen=True)
class HDLCoefficients(BiomarkerCoefficients):
    CARDIO_COEF: float = 0.15
    CARDIO_REFERENCE_MET: float = 1000.0
    STRENGTH_COEF: float = 0.20
    STRENGTH_REFERENCE_MET: float = 450.0
    FIBER_COEF: float = 0.15
    SATURATED_FAT_COEF: float = -0.12
    SFA_PPT_CONVERSION: float = 100.0
    MUSCLE_MASS_COEF: float = 0.02
    BMI_LINEAR_COEF: float = -0.15
    BMI_LINEAR_REFERENCE: float = 22.0
    AGE_COEF: float = -0.08
    AGE_THRESHOLD: float = 35.0
    FEMALE_BONUS: float = 0.2
    AGE_NORM_INTERCEPT: float = 55.0
    AGE_NORM_SLOPE: float = -0.15
    AGE_NORM_THRESHOLD: float = 30.0
    REVERSION_COEF: float = 0.03
    CHANGE_MIN: float = -0.50
    CHANGE_MAX: float = 0.3


@dataclass(frozen=True)
class TotalCholesterolCoefficients(BiomarkerCoefficients):
    SATURATED_FAT_COEF: float = 1.5
    FIBER_COEF: float = -0.5
    ACTIVITY_COEF: float = -0.3
    ACTIVITY_REFERENCE_MET: float = 1000.0
    SFA_PPT_CONVERSION: float = 100.0
    CHANGE_MIN: float = -3.0
    CHANGE_MAX: float = 3.0


@dataclass(frozen=True)
class SBPCoefficients(BiomarkerCoefficients):
    BMI_DELTA_COEF: float = 0.1
    BMI_ABSOLUTE_COEF: float = 0.6
    BMI_ABSOLUTE_THRESHOLD: float = 25.0
    ALCOHOL_COEF: float = 0.01
    ALCOHOL_THRESHOLD_G_PER_DAY: float = 10.0
    SODIUM_COEF: float = 0.05
    SODIUM_REFERENCE_MG: float = 100.0
    STRESS_COEF: float = 0.15
    SLEEP_DEFICIT_COEF: float = 0.25
    SLEEP_THRESHOLD_HOURS: float = 6.0
    AGE_COEF: float = 0.2
    AGE_THRESHOLD: float = 35.0
    REVERSION_COEF: float = 0.1
    AGE_NORM_SBP_INTERCEPT: float = 120.0
    AGE_NORM_SBP_SLOPE: float = 0.6
    AGE_NORM_THRESHOLD: float = 30.0
    CHANGE_MIN: float = -4.0
    CHANGE_MAX: float = 2.0


@dataclass(frozen=True)
class HbA1cCoefficients(BiomarkerCoefficients):
    BASE_DRIFT: float = 0.03
    REVERSION_COEF: float = 0.005
    AGE_NORM_INTERCEPT: float = 5.2
    AGE_NORM_SLOPE: float = 0.045
    AGE_NORM_THRESHOLD: float = 30.0
    SIMPLE_CARBS_COEF: float = 0.05
    SIMPLE_CARBS_REFERENCE_G: float = 10.0
    BMI_CHANGE_COEF: float = 0.12
    FIBER_COEF: float = -0.001
    ACTIVITY_COEF: float = -0.003
    ACTIVITY_REFERENCE_MET: float = 1000.0
    MBI_COEF: float = 0.004
    MUSCLE_MASS_COEF: float = -0.0005
    SLEEP_DEFICIT_COEF: float = 0.004
    SLEEP_THRESHOLD_HOURS: float = 6.0
    CHANGE_MIN: float = -0.1
    CHANGE_MAX: float = 0.05


@dataclass(frozen=True)
class DemographicCoefficients:
    """Demographic distribution parameters."""

    AGE_MEAN: float = 45
    AGE_STD: float = 12
    HEIGHT_MALE_MEAN: float = 178
    HEIGHT_MALE_STD: float = 7
    HEIGHT_FEMALE_MEAN: float = 163
    HEIGHT_FEMALE_STD: float = 6
    HDL_MALE_MEAN: float = 48
    HDL_MALE_STD: float = 9
    HDL_FEMALE_MEAN: float = 58
    HDL_FEMALE_STD: float = 10
    HDL_MIN: float = 15
    HDL_MAX: float = 100
    TC_MEAN: float = 200
    TC_STD: float = 25
    TC_MIN: float = 120
    TC_MAX: float = 350
    TC_GENETIC_FACTOR: float = 0.8
    SBP_INTERCEPT: float = 115
    SBP_AGE_SLOPE: float = 0.5
    SBP_NOISE_STD: float = 8
    SBP_MIN: float = 80
    SBP_MAX: float = 200
    HBA1C_MEAN: float = 5.4
    HBA1C_STD: float = 0.3
    HBA1C_GEN_MIN: float = 4.0
    HBA1C_GEN_MAX: float = 6.5
    HBA1C_FINAL_MIN: float = 3.5
    HBA1C_FINAL_MAX: float = 12.0
    BMI_MEAN: float = 28.5
    BMI_STD: float = 6.0
    BMI_MIN: float = 16
    BMI_MAX: float = 50
    LIPID_CORRELATION: float = 0.3
    LIPID_CLIP: float = 2


@dataclass(frozen=True)
class MuscleCoefficients:
    """Muscle mass distribution and metabolism coefficients."""

    MUSCLE_MALE_MEAN: float = 12
    MUSCLE_MALE_STD: float = 3
    MUSCLE_FEMALE_MEAN: float = 8
    MUSCLE_FEMALE_STD: float = 2.5
    MUSCLE_MIN: float = 0
    MUSCLE_MAX: float = 20
    BMR_CORRECTION: float = 8.5
    TYPICAL_RATIO: float = 0.30
    MM_TO_WEIGHT_KG: float = 0.5
    PROTEIN_REQUIREMENT: float = 1.6
    PROTEIN_DEFICIT: float = 0.8
    SLEEP_DEFICIT_HOURS: float = 6.0


@dataclass(frozen=True)
class LifestyleCoefficients:
    """Lifestyle simulation constants."""

    ALCOHOL_NON_DRINKER_PROB: float = 0.3
    SMOKING_PROB: float = 0.25
    STRESS_EVENT_PROB: float = 0.35
    STRESS_BOOST_MAIN: float = 3.0
    STRESS_BOOST_DECAY: float = 1.5
    SLEEP_DROP: float = 1.0
    ALCOHOL_BOOST_FACTOR: float = 0.2
    WEIGHT_DRIFT_MEAN: float = 0.6
    WEIGHT_DRIFT_STD: float = 3.0
    REVERSION_COEF: float = 0.0015
    APPETITE_SUPPRESSION_BMI_THRESHOLD: float = 30
    APPETITE_SUPPRESSION_COEF: float = 0.03
    MUSCLE_DRIFT_MEAN: float = -0.05
    MUSCLE_DRIFT_STD: float = 0.1
    PROTEIN_TARGET: float = 0.30
    FAT_TARGET: float = 0.30
    CARB_TARGET: float = 0.40

    # NEW: Correlation matrix for OU processes
    OU_CORRELATION_MATRIX: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "alcohol": {"cardio": -0.1, "smoking": 0.2},
            "cardio": {"sleep": 0.1},
            "strength": {"sleep": 0.1},
            "stress": {"sleep": -0.15, "alcohol": 0.1, "smoking": 0.1},
        }
    )


@dataclass(frozen=True)
class GeneticRiskCoefficients:
    CVD: float = 0.4
    DIABETES: float = 0.3
    STROKE: float = 0.3
    NAFLD: float = 0.25
    COLORECTAL: float = 0.35
    CIRRHOSIS: float = 0.4


@dataclass(frozen=True)
class MeasurementNoise:
    SBP: float = 5.0
    HDL: float = 5.0
    TOTAL_CHOLESTEROL: float = 10.0
    HBA1C: float = 0.2
    WEIGHT: float = 1.5


@dataclass(frozen=True)
class ReferenceRanges:
    """Reference values from NHANES, Framingham, WHO for biomarker validation."""

    sbp_mmhg: Dict[str, Dict] = field(
        default_factory=lambda: {
            "name": "Systolic Blood Pressure (mmHg)",
            "source": "NHANES 2017-2020, CDC",
            "age_40_49": {"mean": 122, "std": 15, "p5": 98, "p95": 148},
            "age_50_59": {"mean": 128, "std": 16, "p5": 102, "p95": 155},
            "age_60_69": {"mean": 135, "std": 17, "p5": 108, "p95": 163},
            "delta_20yr": {"mean": 15, "min": 10, "max": 20},
            "unit": "mmHg",
        }
    )
    hba1c_percent: Dict[str, Dict] = field(
        default_factory=lambda: {
            "name": "Glycated Hemoglobin (%)",
            "source": "CDC Diabetes Report 2020",
            "age_40_49": {"mean": 5.6, "std": 0.4, "p5": 5.0, "p95": 6.3},
            "age_50_59": {"mean": 5.7, "std": 0.5, "p5": 5.0, "p95": 6.5},
            "age_60_69": {"mean": 5.8, "std": 0.5, "p5": 5.1, "p95": 6.6},
            "delta_20yr": {"mean": 0.4, "min": 0.2, "max": 0.6},
            "unit": "%",
        }
    )
    hdl_mgdl: Dict[str, Dict] = field(
        default_factory=lambda: {
            "name": "HDL Cholesterol (mg/dL)",
            "source": "NHANES 2017-2020",
            "male": {"mean": 45, "std": 12, "p5": 28, "p95": 68},
            "female": {"mean": 58, "std": 14, "p5": 38, "p95": 85},
            "delta_20yr": {"mean": -7, "min": -10, "max": -2},
            "unit": "mg/dL",
        }
    )
    total_cholesterol_mgdl: Dict[str, Dict] = field(
        default_factory=lambda: {
            "name": "Total Cholesterol (mg/dL)",
            "source": "NHANES 2017-2020",
            "age_40_49": {"mean": 205, "std": 35, "p5": 150, "p95": 265},
            "age_50_59": {"mean": 210, "std": 38, "p5": 155, "p95": 275},
            "age_60_69": {"mean": 200, "std": 40, "p5": 145, "p95": 270},
            "delta_20yr": {"mean": 0, "min": -10, "max": 10},
            "unit": "mg/dL",
        }
    )
    bmi: Dict[str, Dict] = field(
        default_factory=lambda: {
            "name": "Body Mass Index (kg/m^2)",
            "source": "NHANES 2017-2020, CDC",
            "age_40_49": {"mean": 29.0, "std": 5.5, "p5": 21.5, "p95": 39.0},
            "age_50_59": {"mean": 29.5, "std": 5.8, "p5": 21.8, "p95": 40.2},
            "age_60_69": {"mean": 28.8, "std": 5.5, "p5": 21.2, "p95": 39.5},
            "delta_20yr": {"mean": 2.0, "min": 0.5, "max": 4.0},
            "unit": "kg/m^2",
        }
    )
    weight_kg: Dict[str, Dict] = field(
        default_factory=lambda: {
            "name": "Weight (kg)",
            "source": "NHANES 2017-2020",
            "male": {"mean": 90, "std": 18, "p5": 65, "p95": 125},
            "female": {"mean": 77, "std": 16, "p5": 55, "p95": 108},
            "delta_20yr": {"mean": 3, "min": 0, "max": 8},
            "unit": "kg",
        }
    )


@dataclass(frozen=True)
class ScientificCoefficients:
    """Container for all scientific coefficients."""

    hdl: HDLCoefficients = field(default_factory=HDLCoefficients)
    tc: TotalCholesterolCoefficients = field(
        default_factory=TotalCholesterolCoefficients
    )
    sbp: SBPCoefficients = field(default_factory=SBPCoefficients)
    hba1c: HbA1cCoefficients = field(default_factory=HbA1cCoefficients)
    demo: DemographicCoefficients = field(default_factory=DemographicCoefficients)
    muscle: MuscleCoefficients = field(default_factory=MuscleCoefficients)
    lifestyle: LifestyleCoefficients = field(default_factory=LifestyleCoefficients)
    genetic_risk: GeneticRiskCoefficients = field(
        default_factory=GeneticRiskCoefficients
    )
    noise: MeasurementNoise = field(default_factory=MeasurementNoise)
    ref: ReferenceRanges = field(default_factory=ReferenceRanges)

    def __post_init__(self):
        if self.hdl.CHANGE_MIN > self.hdl.CHANGE_MAX:
            logger.warning("HDL change limits are inconsistent: min > max")
        if self.sbp.CHANGE_MIN > self.sbp.CHANGE_MAX:
            logger.warning("SBP change limits are inconsistent: min > max")
        if self.hba1c.CHANGE_MIN > self.hba1c.CHANGE_MAX:
            logger.warning("HbA1c change limits are inconsistent: min > max")

    @property
    def genetic_risk_std(self) -> Dict[str, float]:
        return {
            "cvd": self.genetic_risk.CVD,
            "diabetes": self.genetic_risk.DIABETES,
            "stroke": self.genetic_risk.STROKE,
            "nafld": self.genetic_risk.NAFLD,
            "colorectal": self.genetic_risk.COLORECTAL,
            "cirrhosis": self.genetic_risk.CIRRHOSIS,
        }

    @property
    def to_dict(self) -> Dict:
        result = {}
        for name, field_info in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if hasattr(value, "__dataclass_fields__"):
                result[name] = {
                    k: getattr(value, k) for k in value.__dataclass_fields__
                }
            else:
                result[name] = value
        return result


DEFAULT_COEFFICIENTS = ScientificCoefficients()
