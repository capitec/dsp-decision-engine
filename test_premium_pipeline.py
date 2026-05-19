"""
Insurance premium calculation pipeline — first-time DX exploration.

Steps:
1. Define a Pydantic config with base_rate and age_factor
2. Write base_premium(age, config) and risk_loading(claim_count, base_premium)
3. Generate module, instantiate with specific config values
4. Execute on test data, print results
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from pydantic import BaseModel
from decider.modules.functional import generate_from_functions


# ── Step 1: Config model ──────────────────────────────────────────────────────

class PremiumConfig(BaseModel):
    base_rate: float
    age_factor: float


# ── Step 2 & 3: Functions ────────────────────────────────────────────────────

def base_premium(age: pl.Expr, config: PremiumConfig) -> pl.Expr:
    """Linear age-based premium: age * age_factor + base_rate."""
    return age * config.age_factor + config.base_rate


def risk_loading(claim_count: pl.Expr, base_premium: pl.Expr) -> pl.Expr:
    """Add a flat £50 loading per prior claim."""
    return base_premium + claim_count * 50


# ── Step 4: Generate module and instantiate with config values ────────────────

print("=" * 60)
print("Generating PremiumModule from functions...")
PremiumModule = generate_from_functions("premium", base_premium, risk_loading)
print(f"  Generated class: {PremiumModule}")
print(f"  MRO: {[c.__name__ for c in PremiumModule.__mro__]}")

# Instantiate with concrete config values
print("\nInstantiating with base_rate=200.0, age_factor=15.0 ...")
module = PremiumModule(name="north_polrs_premium", base_rate=200.0, age_factor=15.0)
print(f"  Module: {module}")
print(f"  base_rate={module.base_rate}, age_factor={module.age_factor}")

# Inspect wired inputs/outputs
print(f"\n  input_names : {module.input_names}")
print(f"  output_names: {module.output_names}")

# ── Step 5: Test data ────────────────────────────────────────────────────────

df = pl.DataFrame({
    "policy_id":   ["POL-001", "POL-002", "POL-003", "POL-004"],
    "age":         [25,        35,        50,        65],
    "claim_count": [0,         1,         0,         3],
})
print(f"\nInput DataFrame:\n{df}")

# ── Step 6: Execute ──────────────────────────────────────────────────────────

print("\nExecuting module (lazy=False) ...")
result = module.execute({"input": df}, lazy=False)

print(f"\nResult DataFrame:\n{result}")

# ── Step 7: Spot-check arithmetic ────────────────────────────────────────────
# POL-001: age=25, claims=0  -> base_premium = 25*15 + 200 = 575  -> risk_loading = 575 + 0*50 = 575
# POL-002: age=35, claims=1  -> base_premium = 35*15 + 200 = 725  -> risk_loading = 725 + 50 = 775
# POL-003: age=50, claims=0  -> base_premium = 50*15 + 200 = 950  -> risk_loading = 950 + 0 = 950
# POL-004: age=65, claims=3  -> base_premium = 65*15 + 200 = 1175 -> risk_loading = 1175 + 150 = 1325

expected_base    = [575.0,  725.0,  950.0, 1175.0]
expected_loading = [575.0,  775.0,  950.0, 1325.0]

assert result["base_premium"].to_list() == expected_base, \
    f"base_premium mismatch: {result['base_premium'].to_list()}"
assert result["risk_loading"].to_list() == expected_loading, \
    f"risk_loading mismatch: {result['risk_loading'].to_list()}"

print("\nAll assertions passed.")
print("=" * 60)
