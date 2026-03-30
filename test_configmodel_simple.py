import typing as t
from pydantic import BaseModel, RootModel, model_serializer, FieldSerializationInfo, SerializerFunctionWrapHandler, model_validator, ConfigDict, PrivateAttr, Discriminator, Tag
from dataclasses import dataclass
from decider.modules.core import BaseModule
from decider.modules import GraphModule

T = t.TypeVar("T")
PARAMETERS = {
    "value": 5,
    "str_value": "from_params",
    "pos_x": 100.5,
    "item_a": "list_item",
    "dict_c": 999
}

class Parameter(BaseModel, t.Generic[T]):
    default: T
    key: str

    @model_serializer(mode='wrap')
    def serialize_model(
        self, handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> dict[str, object]:
        global PARAMETERS
        if isinstance(info.context, dict) and info.context.get("decider.config.parameterize", False):
            return PARAMETERS.get(self.key, self.default)

        return handler(self)


TRawModuleConfig = t.Dict[str, 'TModuleConfigValue'] | t.List['TModuleConfigValue'] | t.Any

TModuleConfigValue = t.Annotated[t.Union[
   t.Annotated[TRawModuleConfig, Tag('generic')],
   t.Annotated[Parameter, Tag('parameter')]
], Discriminator(lambda x: 'parameter' if isinstance(x,Parameter) else 'generic')]



class ModuleConfig(BaseModel):
    root: TModuleConfigValue

class ConfigSelector:
    def __init__(
        self, 
        config_module: "ConfigModule", 
        path: t.Tuple[t.Union[str, int], ...], 
    ):
        self.config_module = config_module
        self.path = path
    
    def __getattr__(self, name: str):
        if name.startswith('_') or name in ('config_module', 'path', 'parent', 'parent_key', 'set_as_parameter'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        return ConfigSelector(self.config_module, self.path + (name,), None, None)
    
    def __getitem__(self, key: t.Union[int, str]):
        return ConfigSelector(self.config_module, self.path + (key,), None, None)
    

    def set_as_parameter(self, key: str):
        """Mark this path as a parameter"""
        self.config_module.config_overrides.append((self.path, key))
        self.config_module._set_parameter(self.path)

class ConfigModule(BaseModule):
    model_config = ConfigDict(extra='allow')
    type: t.Literal["config"]
    inner_module_type: str
    config_overrides: t.List[t.Tuple[t.Tuple[t.Union[int,str],...], str]] = []

    _module_config: ModuleConfig = PrivateAttr(default=None)

    @classmethod
    def from_module(cls, module: BaseModule, name: str = None) -> "ConfigModule":
        """Create a ConfigModule from an existing BaseModule instance
        
        Args:
            module: The module to wrap in a ConfigModule
            name: Optional name for the ConfigModule. If not provided, uses module.name
            
        Returns:
            ConfigModule wrapping the provided module
        """
        # Convert module to dict, excluding BaseModule fields
        module_dict = module.model_dump(mode='json')
        inner_type = module_dict.pop('type')
        module_name = module_dict.pop('name', None)
        
        return cls(
            type='config',
            name=name or module_name or 'config_wrapper',
            inner_module_type=inner_type,
            **module_dict
        )

    @property
    def config(self) -> "ConfigSelector":
        return ConfigSelector(self, ())

    @classmethod
    def _construct_config(cls, parent, parent_key, root, path):
        if len(path) == 0:
            # Here we dont convert because we assume its already converted in the dict
            parent[parent_key] = Parameter(root)
            return
        _construct_config(..)

    def _get_injected_config(self) -> ModuleConfig:
        extra_data = self.model_extra or {}
        base_data = {"type": self.inner_module_type, "name": self.name}
        raw_config = base_data | extra_data
        for path in self.config_overrides:
            self._construct_config(...)


    def _rebuild_module_config(self):
        """Rebuild the module config with Parameter types at specified paths"""
        extra_data = self.model_extra or {}
        base_data = {"type": self.inner_module_type, "name": self.name}
        raw_config = base_data | extra_data
        
        # Inject Parameter instances at the specified override paths in the data
        self._module_config = ModuleConfig.model_validate({"root": self._inject_parameter_instances(raw_config)})
    


    

    @model_validator(mode='after')
    def initialise_internal_components(self):
        # Look up the type from GraphModule.root union based on inner_module_type
        self._rebuild_module_config()
        return self
    

    @model_serializer(mode='wrap')
    def serialize_model(
        self, handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> dict[str, object]:
        if isinstance(info.context, dict) and info.context.get("decider.config.parameterize", False):
            return self._module_config.root
        
        serialized = handler(self)
        root_data = self._module_config.model_dump(mode=info.mode)['root']
        
        return {
            **root_data,
            "type": serialized["type"],
            "inner_module_type": serialized["inner_module_type"],
            "config_overrides": serialized["config_overrides"],
        }

    def get_parameterized_module(self) -> GraphModule:
        root_dict = self._module_config.model_dump(context={"decider.config.parameterize": True})['root']
        return GraphModule.model_validate({"root": root_dict})
    
    def expand_nodes(self):
        return []


if __name__ == "__main__":
    from decider.modules.primitives import MapperModule
    
    print("=" * 50)
    print("Starting Simplified ConfigModule tests")
    print("=" * 50)
    print()
    
    # Test 1: from_module with simple field parameterization
    print("=" * 50)
    print("TEST 1: from_module + simple field")
    print("=" * 50)
    
    base_module = MapperModule(name="test_mapper", modules=[], mappings={})
    c1 = ConfigModule.from_module(base_module)
    
    c1.config.name.set_as_parameter("str_value")
    c1._rebuild_module_config()
    
    dump1 = c1._module_config.model_dump()
    print(f"name value: {dump1['root']['name']}")
    
    assert isinstance(dump1['root']['name'], dict)
    assert dump1['root']['name']['default'] == "test_mapper"
    assert dump1['root']['name']['key'] == "str_value"
    print("✓ from_module + simple field works")
    print()
    
    # Test 2: Wildcard dict access
    print("=" * 50)
    print("TEST 2: Wildcard dict access")
    print("=" * 50)
    
    base_module2 = MapperModule(
        name="test_mapper2",
        modules=[],
        mappings={"key1": {"in1": ("m1", "o1"), "in2": ("m2", "o2")}}
    )
    c2 = ConfigModule.from_module(base_module2)
    
    c2.config.mappings['*']['*'].set_as_parameter("value")
    c2._rebuild_module_config()
    
    dump2 = c2._module_config.model_dump()
    print(f"mappings['key1']['in1']: {dump2['root']['mappings']['key1']['in1']}")
    print(f"mappings['key1']['in2']: {dump2['root']['mappings']['key1']['in2']}")
    
    assert dump2['root']['mappings']['key1']['in1']['default'] == ("m1", "o1")
    assert dump2['root']['mappings']['key1']['in1']['key'] == "value"
    assert dump2['root']['mappings']['key1']['in2']['default'] == ("m2", "o2")
    assert dump2['root']['mappings']['key1']['in2']['key'] == "value"
    print("✓ Wildcard dict access works")
    print()
    
    # Test 3: get_parameterized_module
    print("=" * 50)
    print("TEST 3: get_parameterized_module")
    print("=" * 50)
    
    built = c1.get_parameterized_module()
    print(f"Built module type: {type(built)}")
    built_dict = built.model_dump()
    print(f"name after parameterization: {built_dict['root']['name']}")
    
    assert built_dict['root']['name'] == "from_params"
    print("✓ get_parameterized_module works")
    print()
    
    print("=" * 50)
    print("All simplified tests passed! ✓✓✓")
    print("=" * 50)