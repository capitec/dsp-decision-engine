"""Patient risk stratification pipeline using the decider framework.

Built by reading docstrings + source files cold. See DX report at the bottom.

Pipeline:
  1. JoinModule: join patients (patient_id, age, bmi) with labs (patient_id, glucose, systolic_bp)
  2. generate_from_functions: compute metabolic_risk, hypertension_flag, overall_risk
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.primitives.join import JoinModule
from decider.modules.functional import generate_from_functions


# ── Step 1: Join module ───────────────────────────────────────────────────────
# From JoinModule docstring: left/right are frame names in the input dict,
# output_frame is what the joined result is stored as downstream.

join_demographics_labs = JoinModule(
    name="join_demographics_labs",
    left="patients",
    right="labs",
    on="patient_id",
    how="inner",
    output_frame="patient_labs",
)

# ── Step 2: Risk expression module ────────────────────────────────────────────
# From generate_from_functions docstring: function name → output column name,
# parameter name → input column name.
# QUESTION GOING IN: after a join, what frame do the expressions target?
# ANSWER (from pipeline.py:104): the pipeline aliases the previous frame as "input",
# so expressions just read columns by name — no special wiring needed.

def metabolic_risk(glucose: pl.Expr, bmi: pl.Expr) -> pl.Expr:
    """Blended metabolic risk: 50% glucose elevation + 50% BMI elevation, each clipped 0–1."""
    glucose_component = (glucose / 100 - 1).clip(0, 1) * 0.5
    bmi_component = (bmi / 30 - 1).clip(0, 1) * 0.5
    return glucose_component + bmi_component


def hypertension_flag(systolic_bp: pl.Expr) -> pl.Expr:
    """1 if systolic_bp > 130 mmHg, else 0."""
    return (systolic_bp > 130).cast(pl.Int8)


def overall_risk(metabolic_risk: pl.Expr, hypertension_flag: pl.Expr) -> pl.Expr:
    """Weighted composite risk score (metabolic 60%, hypertension 40%)."""
    return metabolic_risk * 0.6 + hypertension_flag * 0.4


# Note: metabolic_risk and hypertension_flag are both function names AND parameter
# names in overall_risk — the framework wires them as internal node dependencies
# automatically (generate_from_functions docstring, rule 2a).

RiskScorer = generate_from_functions(
    "risk_scorer",
    metabolic_risk,
    hypertension_flag,
    overall_risk,
)
risk_scorer = RiskScorer(name="risk_scorer")

# ── Step 3: Compose pipeline ──────────────────────────────────────────────────
# From Pipeline docstring: modules compose with |

pipeline = join_demographics_labs | risk_scorer


# ── Step 4: Test data ─────────────────────────────────────────────────────────

patients = pl.DataFrame({
    "patient_id": [1, 2, 3, 4],
    "age":        [45, 62, 38, 71],
    "bmi":        [22.0, 34.5, 28.0, 41.2],   # 22=healthy, 34.5=obese, 28=overweight, 41.2=severely obese
})

labs = pl.DataFrame({
    "patient_id": [1, 2, 3, 4],
    "glucose":    [88.0, 145.0, 102.0, 178.0],  # 88=low-normal, 145=elevated, 102=borderline, 178=high
    "systolic_bp":[118, 142, 125, 158],           # 118=normal, 142=hypertensive, 125=normal, 158=hypertensive
})


if __name__ == "__main__":
    print("=" * 60)
    print("Patient Risk Stratification Pipeline")
    print("=" * 60)

    print("\n[INPUT] Patients:")
    print(patients)
    print("\n[INPUT] Labs:")
    print(labs)

    # Execute — pass named frames; "input" key not needed for JoinModule (it reads
    # left/right by name). Pipeline.execute auto-converts DataFrame → LazyFrame.
    result = pipeline.execute(
        {"patients": patients, "labs": labs},
        lazy=False,   # collect immediately for readable output
    )

    print("\n[RESULT] Risk scores:")
    print(result)

    # Spot-check patient 1 (healthy baseline):
    # glucose=88 → (88/100-1)=-0.12 → clip→0 ; bmi=22 → (22/30-1)=-0.27 → clip→0
    # metabolic_risk = 0 ; hypertension_flag = 0 (118 ≤ 130) ; overall_risk = 0
    row1 = result.filter(pl.col("patient_id") == 1)
    assert abs(row1["metabolic_risk"][0] - 0.0) < 1e-6, "Patient 1 metabolic_risk should be 0"
    assert row1["hypertension_flag"][0] == 0, "Patient 1 hypertension_flag should be 0"
    assert abs(row1["overall_risk"][0] - 0.0) < 1e-6, "Patient 1 overall_risk should be 0"

    # Patient 2: glucose=145 → (1.45-1)=0.45*0.5=0.225 ; bmi=34.5 → (34.5/30-1)=0.15*0.5=0.075
    # metabolic_risk=0.3 ; hypertension_flag=1 (142>130) ; overall_risk=0.3*0.6+1*0.4=0.58
    row2 = result.filter(pl.col("patient_id") == 2)
    assert abs(row2["metabolic_risk"][0] - 0.3) < 1e-6, f"Patient 2 metabolic_risk: got {row2['metabolic_risk'][0]}, expected 0.3"
    assert row2["hypertension_flag"][0] == 1, "Patient 2 hypertension_flag should be 1"
    assert abs(row2["overall_risk"][0] - 0.58) < 1e-6, f"Patient 2 overall_risk: got {row2['overall_risk'][0]}, expected 0.58"

    print("\n[OK] All assertions passed.")

    # Pretty risk bands
    print("\n[SUMMARY] Risk stratification:")
    bands = result.with_columns(
        pl.when(pl.col("overall_risk") < 0.2).then(pl.lit("LOW"))
          .when(pl.col("overall_risk") < 0.5).then(pl.lit("MEDIUM"))
          .otherwise(pl.lit("HIGH"))
          .alias("risk_band")
    ).select(["patient_id", "age", "metabolic_risk", "hypertension_flag", "overall_risk", "risk_band"])
    print(bands)
