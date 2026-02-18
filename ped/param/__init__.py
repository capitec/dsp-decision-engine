# TODO not sure where is best to do this
# I think here makes sense because config_source_provider should be used wherever we use ped.param

def _register_resolvers():
    from .resolver import resolve_parameter
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("ped.param", resolve_parameter)

_register_resolvers()