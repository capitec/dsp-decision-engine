import polars as pl
from pydantic import BaseModel
from decider.modules.functional import generate_from_functions
from decider.modules import register_graph_module

$user_code

$class_name = generate_from_functions(
    "$type_id",
    $function_names,
)

register_graph_module($class_name)
