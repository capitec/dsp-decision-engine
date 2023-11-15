def dangerously_skip_type_validation(fn):
    fn._skip_type_validation_=True
    return fn

def require_inject_function_map(fn):
    fn._inject_function_map_=True
    return fn