"""
Loan scoring project — generates config artifacts and verifies the full
model-creation → config-save → serve round-trip.

Run from the project root:
    python projects/loan_scoring/generate.py

What it does:
    1. Calls initialize_decider() with the project's extension path so that
       CreditScorer (defined in decider_extensions/loan_scoring/) is registered
       into GraphModule before any config is read or written.
    2. Builds a scorer instance with tunable config fields.
    3. Saves it as a versioned JSON config under projects/loan_scoring/configs/.
    4. Reads the config back via a fresh config manager and reconstructs the
       module — proving round-trip fidelity through the same path the server uses.
    5. Runs the pipeline on a test batch and asserts correctness.
    6. Prints the env-var commands needed to point a server at this project.
"""

import sys
import os
import asyncio
import json
import polars as pl

# Make sure project root is on the path when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

EXTENSIONS_DIR = os.path.join(PROJECT_DIR, "decider_extensions")

from decider.initialization import initialize_decider
from decider.config.file import JsonFileConfigManager
from decider.modules import GraphModule

CONFIGS_DIR = os.path.join(PROJECT_DIR, "configs")
ROOT_KEY = "main"

BATCH = pl.DataFrame({
    "applicant_id": ["A001", "A002", "A003"],
    "debt":         [25_000.0, 5_000.0,  80_000.0],
    "income":       [50_000.0, 60_000.0, 90_000.0],
    "credit_used":  [4_000.0,  500.0,   18_000.0],
    "credit_limit": [10_000.0, 5_000.0, 20_000.0],
})


async def main():
    print("=" * 60)
    print("LOAN SCORING — config generation & serve round-trip")
    print("=" * 60)

    # ── Step 1: register extensions (same call the servers make) ─────────────
    initialize_decider(extension_path=EXTENSIONS_DIR)
    # CreditScorer is now registered in GraphModule
    from loan_scoring import CreditScorer
    print(f"\n[1] Extension loaded — CreditScorer type={CreditScorer._CLASS_TYPE_IDENTIFIER!r}")

    # ── Step 2: build module with tunable config values ───────────────────────
    scorer = CreditScorer(
        name="credit_scorer",
        dti_weight=200.0,
        utilization_weight=100.0,
        score_base=800.0,
    )
    print(f"[2] Module built: type={scorer.type!r}  name={scorer.name!r}")

    # ── Step 3: save to versioned JSON config ─────────────────────────────────
    config_manager = JsonFileConfigManager(basepath=CONFIGS_DIR)
    versioned = await scorer.asave(ROOT_KEY, config_manager)
    await config_manager.save_version(overwrite=True)
    written_path = os.path.join(CONFIGS_DIR, str(versioned.version), f"{ROOT_KEY}.json")
    print(f"\n[3] Saved version {versioned.version} to: {written_path}")

    with open(written_path) as f:
        on_disk = json.load(f)
    print(f"\n[4] Config on disk:")
    print(json.dumps(on_disk, indent=2))

    # ── Step 4: round-trip — fresh manager, reconstruct from disk ─────────────
    fresh_manager = JsonFileConfigManager(basepath=CONFIGS_DIR)
    loaded = await fresh_manager.get_latest()
    print(f"\n[5] Loaded version: {loaded.version}")

    module = GraphModule.model_validate(loaded.config[ROOT_KEY]).root
    print(f"    Reconstructed: type={module.type!r}  dti_weight={module.dti_weight}  "
          f"utilization_weight={module.utilization_weight}  score_base={module.score_base}")

    # ── Step 5: execute and verify ────────────────────────────────────────────
    result = module({"input": BATCH})
    print(f"\n[6] Pipeline output:")
    print(result.select(["applicant_id", "dti_ratio", "utilization_rate", "credit_score_estimate"]))

    # A001: dti=0.5, util=0.4  → 800 - 0.5*200 - 0.4*100 = 660.0
    a001 = result.filter(pl.col("applicant_id") == "A001")["credit_score_estimate"][0]
    assert abs(a001 - 660.0) < 1e-4, f"A001: expected 660.0, got {a001}"
    # A002: dti≈0.0833, util=0.1 → 800 - 16.67 - 10 ≈ 773.33
    a002 = result.filter(pl.col("applicant_id") == "A002")["credit_score_estimate"][0]
    assert abs(a002 - 773.33) < 0.1, f"A002: expected ~773.33, got {a002}"

    print("\n[OK] All assertions passed.")

    # ── Step 6: serving instructions ──────────────────────────────────────────
    abs_configs = os.path.abspath(CONFIGS_DIR)
    print(f"""
{"=" * 60}
SERVING SETUP
{"=" * 60}
Set these env vars then start either server.  The extension path tells the
server to load loan_scoring/ at startup so CreditScorer is in GraphModule
before any config is parsed.

  export Decider_config__type=file:json
  export Decider_config__basepath={abs_configs}
  export Decider_api__root_module={ROOT_KEY}
  export Decider_ext__extension_path={os.path.abspath(EXTENSIONS_DIR)}

  # Starlette (uvicorn):
  uvicorn decider.serving.servers.starlette:app --host 0.0.0.0 --port 8080

  # Sanic:
  python -m sanic decider.serving.servers.sanic:app --host 0.0.0.0 --port 8080

Test request:
  curl -X POST http://localhost:8080/predict \
       -H 'Content-Type: application/json' \
       -d '{{"debt":[25000],"income":[50000],"credit_used":[4000],"credit_limit":[10000]}}'
""")


if __name__ == "__main__":
    asyncio.run(main())
