import json

from decikit._tooling.mcp._docs import read_doc, save_doc

TREE = {"type": "tree", "name": "risk", "tree": {
    "nodes": [{"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "ratio", "threshold": 0.7}}},
              {"id": "high", "data": {"type": "leaf", "result_idx": 0}}],
    "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
    "output": {"data": [{"risk": 1}], "default": {"risk": 0}, "dtypes": [["risk", "Int64"]]}}}


def test_a_document_that_does_not_build_is_not_saved(tmp_path):
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(TREE))
    broken = json.loads(json.dumps(TREE))
    broken["tree"]["edges"][0]["target"] = "nope"
    result = save_doc(str(path), broken)
    assert not result["saved"] and "unknown node 'nope'" in result["problems"][0]
    assert read_doc(str(path)) == TREE


def test_saving_keeps_the_files_layout(tmp_path):
    for text in (json.dumps(TREE), json.dumps(TREE, indent=1) + "\n", json.dumps(TREE, indent=4)):
        path = tmp_path / "risk.json"
        path.write_text(text)
        assert save_doc(str(path), read_doc(str(path)))["saved"]
        assert path.read_text() == text
