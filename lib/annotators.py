def dangerously_skip_type_validation(fn):
    fn._skip_type_validation_=True
    return fn