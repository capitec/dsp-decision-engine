import typing as t
from hamilton import node
import polars as pl


def create_node_with_mapping(
    func: t.Callable,
    name: str = None,
    input_mapping: t.Optional[t.Dict[str, str]] = None,
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> node.Node:
    """Creates a Hamilton node from a function with parameter mapping and partial application.
    
    Args:
        func: The function to wrap in a node
        input_mapping: Dictionary mapping external parameter names to internal function parameter names
                      e.g., {"variable_name": "input"} maps external "variable_name" to function's "input" param
        partial_kwargs: Dictionary of keyword arguments to partially apply to the function
        
    Returns:
        A Hamilton node with appropriately mapped inputs and proper type information
        
    Example:
        # Function expects parameter 'input', but DAG will provide 'my_variable'
        node = create_node_with_mapping(
            score_variable,
            input_mapping={"my_variable": "input"},
            partial_kwargs={"bins": [...], "default": DefaultBin(...)}
        )
    """
    if partial_kwargs is None:
        partial_kwargs = {}
    if input_mapping is None:
        input_mapping = {}
    
    # Create the original node to get input types
    original_node = node.Node.from_fn(func)
    
    def wrapper_function(**kwargs):
        nonlocal func, input_mapping, partial_kwargs
        # Apply input mapping: map external names to internal parameter names
        mapped_kwargs = {}
        for external_name, value in kwargs.items():
            internal_name = input_mapping.get(external_name, external_name)
            mapped_kwargs[internal_name] = value
        
        # Merge with partial kwargs (partial kwargs take precedence)
        final_kwargs = {**mapped_kwargs, **partial_kwargs}
        
        return func(**final_kwargs)
    
    # Validate partial kwargs against the original function's parameters
    function_allows_kwargs = "kwargs" in original_node.input_types
    
    if not function_allows_kwargs:
        for kwarg in partial_kwargs:
            if kwarg not in original_node.input_types:
                raise ValueError(f"Partial argument '{kwarg}' is not a valid parameter of the function.")

    # Build new input types based on mapping
    new_input_types = {}
        
    for external_name, internal_name in input_mapping.items():
        if internal_name in original_node.input_types:
            # Map the type from internal to external parameter
            new_input_types[external_name] = original_node.input_types[internal_name]
        else:
            if not function_allows_kwargs:
                raise ValueError(f"Original function must contain parameters in the mapping or have a **kwargs parameter to use for unmapped parameters. Missing parameter: {internal_name}")
            new_input_types[external_name] = original_node.input_types["kwargs"]

    
    # Add any remaining input types that weren't mapped and aren't in partial_kwargs
    for param_name, param_info in original_node.input_types.items():
        if param_name not in partial_kwargs and param_name not in input_mapping.values() and param_name != "kwargs":
            new_input_types[param_name] = param_info

    # If no input mapping was needed, just apply partial kwargs
    if not input_mapping and not partial_kwargs:
        if name is not None:
            return original_node.copy_with(name=name)
        return original_node
    
    return original_node.copy_with(
        callabl=wrapper_function,
        input_types=new_input_types,
        name=name or original_node.name
    )