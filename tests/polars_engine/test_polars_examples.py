"""Every example, built from its latest config version, gives the fused answer on the polars engine."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
PROJECTS = sorted(p.name for p in EXAMPLES.iterdir() if (p / "README.md").is_file())

# One process per project: the example packages register config types that other tests in this process also claim.
SCRIPT = """
import json, sys, warnings
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from decikit import PolarsEngine
from decikit.config import JsonFileStore
from decikit.exceptions import EngineError
from decikit.serving import RequestHandler
from decikit.serving._parse import coerce_record

warnings.simplefilter("ignore")
package, configs, sample, out = sys.argv[1:]
handler = RequestHandler(JsonFileStore(basepath=configs), package + ".pipeline:build", sample_request=sample)
handler.stage()
handler.activate()
live = handler.module_fn()
exe, params = live.executable, live.params
record = coerce_record(json.loads(Path(sample).read_text()), live.dates)
rows = [record]
for k in (0.5, 0.8, 1.25, 2.0, 3.0):
    row = {key: v * k if isinstance(v, float) else v for key, v in record.items()}
    try:
        exe.run(pl.DataFrame(rows + [row]), params)   # a variation fused itself refuses is no test of the host
    except Exception:
        continue
    rows.append(row)
df = pl.DataFrame(rows * (1 + 8 // len(rows)))
want = exe.run(df, params)


def same(got):
    # A Python-built column with no value is `Null` on fused, and the host types it from its annotation.
    nulls = {c: got.schema[c] for c, t in want.schema.items() if t == pl.Null}
    assert all(got[c].null_count() == got.height for c in nulls)
    assert_frame_equal(got, want.cast(nulls), check_exact=True)


try:
    host = PolarsEngine().bind(exe.step)
    lf = host.run(df.lazy(), params)
except EngineError as e:
    print(json.dumps({"skip": str(e)}))
    sys.exit(0)
same(lf.collect())
pl.Config.set_streaming_chunk_size(3)
same(lf.collect(engine="streaming"))
fmt = "parquet"
try:
    df.write_parquet(Path(out) / "in.parquet")
except pl.exceptions.InvalidOperationError:   # a struct with no fields, which parquet cannot hold and Arrow can
    fmt = "ipc"
    df.write_ipc(Path(out) / "in.ipc")
getattr(host.run(getattr(pl, "scan_" + fmt)(Path(out) / f"in.{fmt}"), params), "sink_" + fmt)(Path(out) / f"out.{fmt}")
same(getattr(pl, "read_" + fmt)(Path(out) / f"out.{fmt}"))
print(json.dumps({"segments": [streamable for _, streamable in host.segments], "rows": df.height, "sink": fmt}))
"""


@pytest.mark.parametrize("project", PROJECTS)
def test_every_example_flow_equals_fused(project, tmp_path):
    root = EXAMPLES / project
    package = next(p.name for p in root.iterdir() if (p / "pipeline.py").is_file())
    path = os.pathsep.join(str(p) for p in sorted(EXAMPLES.iterdir()) if p.is_dir())
    run = subprocess.run(
        [sys.executable, "-c", SCRIPT, package, str(root / "configs"), str(root / "sample_request.json"), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": path}, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    result = json.loads(run.stdout.splitlines()[-1])
    if "skip" in result:
        pytest.skip(result["skip"])
