import typing
import numpy as np
import pandas as pd
from spockflow.core import initialize_spock_module
from spockflow.components.treelite import Tree
from hamilton.function_modifiers import inject, source

results = Tree.from_config("tree")

@inject(
    prioritized_tree_paths=source('results.prioritized_tree_paths'), 
    path_keys=source('results.tree_path_keys')
)
def paths(
    prioritized_tree_paths: np.ndarray, 
    path_keys: typing.List[str]
) ->pd.DataFrame:
    return pd.DataFrame(data=prioritized_tree_paths, columns=path_keys)

initialize_spock_module(
    __name__,
    output_names=[
            "results", "paths"
        ],
    included_modules=[],
)