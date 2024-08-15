import os
import sys
from hamilton.driver import Driver as HDriver
from spockflow.core import Driver

PIPELINE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "test_pipelines"))

def test_nested_import_dag():
    sys.path.insert(0, os.path.join(PIPELINE_DIR,"p01_nested_imports"))
    import main
    dr = Driver({}, main)
    assert len(dr.graph.nodes) == 5