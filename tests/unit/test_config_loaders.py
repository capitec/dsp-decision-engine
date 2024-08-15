import os
import pytest

PIPELINE_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "test_pipelines"))

def test_loading_empty_conf():
    from spockflow.inference.config.loader import empty
    config_loader = empty.EmptyConfigManager()
    assert config_loader.get_latest_version('test') == ''
    assert config_loader.get_config(None, None) == {}

def test_pipeline_discovery():
    from spockflow.inference.util import autodiscover_entrypoints, load_all_entrypoints, register_model_dir
    basic_pipeline_dir = os.path.join(PIPELINE_DIR, "basic")
    register_model_dir(basic_pipeline_dir)
    entrypoint_conf = autodiscover_entrypoints(basic_pipeline_dir)
    assert entrypoint_conf == {'1': ["proc_1"]}
    entrypoints = load_all_entrypoints(entrypoint_conf)
    assert len(entrypoints) == 2
    assert set(entrypoints.keys()) == {"1", "__default__"}

