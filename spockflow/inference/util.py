import typing
from logging import getLogger
from .settings import get_settings

if typing.TYPE_CHECKING:
    from types import ModuleType
    from spockflow.core import Driver
    from spockflow.inference.model_loader import TModuleList, TModuleListMap

logger = getLogger(__name__)


def register_model_dir(model_dir: str):
    """Add model to PYTHON_PATH"""
    import sys

    if model_dir not in sys.path:
        sys.path.insert(1, model_dir)


def load_entrypoint_modules(module_list: typing.List[str]) -> "TModuleList":
    """A function to load a set of modules used in the hamilton DAG"""
    if len(module_list) <= 0:
        raise ValueError(
            "Module list must contain at least one module to be loaded by spock"
        )
    from importlib import import_module

    return [import_module(m) for m in module_list]


def find_default_entrypoint() -> "TModuleList":
    DEFAULT_ENTRYPOINTS = get_settings().default_entrypoints
    for ep_modules in DEFAULT_ENTRYPOINTS:
        try:
            return load_entrypoint_modules(ep_modules)
        except ImportError:
            pass
    raise ValueError("Could not find default entrypoint")


def load_all_entrypoints(
    entrypoint_configuration: typing.Dict[str, typing.List[str]]
) -> "TModuleListMap":
    mods = {k: load_entrypoint_modules(v) for k, v in entrypoint_configuration.items()}
    if "__default__" not in mods:
        try:
            mods["__default__"] = find_default_entrypoint()
        except ValueError as e:
            error_detail = "\nPlease ensure MODEL_PATH is configured correctly and that it contains the correct module or manually configure a default."
            if mods:
                logger.warn(f"Could not load a default entrypoint." + error_detail)
            else:
                raise ValueError("Could not find entrypoint." + error_detail) from e
    return mods


def cast_result_of_model_fn(
    modules: typing.Union["TModuleList", "TModuleListMap", "ModuleType"]
) -> "TModuleListMap":
    """Allow user to return multiple types from the model fn and cast it to the form expected"""
    if isinstance(modules, dict):
        return modules
    if not isinstance(modules, list):
        modules = [modules]
    return {"__default__": modules}


def autodiscover_entrypoints(model_dir: str):
    import os
    from glob import glob

    proc_prefix = get_settings().proc_prefix
    proc_prefix_len = len(proc_prefix)
    proc_modules = (
        os.path.splitext(os.path.split(p)[1])[0]
        for p in glob(os.path.join(model_dir, proc_prefix + "*.py"))
    )
    return {p[proc_prefix_len:]: [p] for p in proc_modules}


# def get_internal_nodes(model: "Driver") -> typing.Set[str]:
#     from hamilton.node import NodeType
#     return {n for n in model.graph.nodes if n.node_role != NodeType.EXTERNAL}


def split_model_inputs(model: "Driver", model_data: dict):
    from hamilton.node import NodeType, Node

    if not get_settings().server_enable_model_overrides:
        return model_data, {}

    input_data = {}
    override_data = {}
    for k, v in model_data.items():
        node: typing.Optional[Node] = model.graph.nodes.get(k)
        if node is not None and node.node_role != NodeType.EXTERNAL:
            override_data[k] = v
        else:
            input_data[k] = v

    return input_data, override_data


def install_requirements(requirements_path: str):
    # This is used from
    # https://github.com/aws/sagemaker-scikit-learn-container/blob/602367b72c30159a12cc4dc2dfc17ab5b338169b/src/sagemaker_sklearn_container/mms_patch/model_server.py#L149
    import sys
    import subprocess

    logger.info("installing packages from requirements.txt...")
    pip_install_cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]

    try:
        subprocess.check_call(pip_install_cmd)
    except subprocess.CalledProcessError:
        logger.error("failed to install required packages, exiting")
        raise ValueError("failed to install required packages")
