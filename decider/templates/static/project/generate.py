"""
$project_title — generates config artifacts and verifies the round-trip.

Run from the project root:
    python $project_dir/generate.py
"""

import sys
import os
import asyncio
import json
import polars as pl

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
    # TODO: replace with a sample batch for your module
    "id": ["row_1", "row_2"],
})


async def main():
    print("=" * 60)
    print("$project_title — config generation & serve round-trip")
    print("=" * 60)

    initialize_decider(extension_path=EXTENSIONS_DIR)
    print("[1] Extensions loaded")

    # TODO: import and build your module here
    # from my_extension import MyModule
    # module = MyModule(name="main")

    config_manager = JsonFileConfigManager(basepath=CONFIGS_DIR)
    # versioned = await module.asave(ROOT_KEY, config_manager)
    # await config_manager.save_version(overwrite=True)
    # print(f"[2] Saved version {versioned.version}")

    abs_configs = os.path.abspath(CONFIGS_DIR)
    print(f"""
{"=" * 60}
SERVING SETUP
{"=" * 60}
  export Decider_config__type=file:json
  export Decider_config__basepath={abs_configs}
  export Decider_api__root_module={ROOT_KEY}
  export Decider_ext__extension_path={os.path.abspath(EXTENSIONS_DIR)}

  uvicorn decider.serving.servers.starlette:app --host 0.0.0.0 --port 8080
""")


if __name__ == "__main__":
    asyncio.run(main())
