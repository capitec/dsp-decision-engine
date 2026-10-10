import json
import math

from decikit._tooling.mcp import Policy, build_server
from tests.mcp.test_mcp import _call

BANDS = {"type": "decision_table", "name": "band", "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
         "rows": [{"lo": -math.inf, "hi": 600.0, "band": "C"}, {"lo": 600.0, "hi": math.inf, "band": "A"}],
         "expression": {"type": "between", "variable": "score", "lower_bound_column": "lo", "upper_bound_column": "hi"},
         "outputs": ["band"], "default": ["C"]}


def test_infinite_bounds_are_spelled_out_in_replies_and_read_back_from_arguments(tmp_path):
    path = tmp_path / "band.json"
    path.write_text(json.dumps(BANDS, indent=2) + "\n")
    before = path.read_text()
    mcp = build_server(Policy())

    view = _call(mcp, "table_view", path=str(path))
    assert view["rows"][0]["lo"] == "-Infinity" and view["rows"][1]["hi"] == "Infinity"

    doc = _call(mcp, "read_config", path=str(path))["doc"]
    assert _call(mcp, "write_config", path=str(path), doc=doc, overwrite=True)["saved"]
    assert path.read_text() == before
