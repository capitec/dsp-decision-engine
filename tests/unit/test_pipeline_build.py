import os
import sys
from importlib import import_module
from spockflow.core import Driver

PIPELINE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "test_pipelines"))
sys.path.insert(0, os.path.join(PIPELINE_DIR))


def test_nested_import_dag():
    main = import_module("p01_nested_imports.main")
    dr = Driver({}, main)
    assert len(dr.graph.nodes) == 7
