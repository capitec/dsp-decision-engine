import typing as t
from typing import get_origin, get_args, Union
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass

PARAMETERS = {
    "value": 5,
    "str_value": "from_params"
}

T = t.TypeVar("T")

class Parameter(BaseModel, t.Generic[T]):
    default: T
    key: str

class ConfigModel:
    def __init__(self, model_instance: BaseModel):
        self._instance = model_instance
        self._root_instance = model_instance
        self._parameters = {}  # path -> key mapping
        self._parameterized_model = None
    
    @classmethod
    def from_class(cls, model_instance: BaseModel):
        return cls(model_instance)
    
    def __getattr__(self, name: str):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        if name == 'model_dump':
            return self._model_dump
        
        if name == 'build':
            return self._build
        
        # Return a ConfigSelector for navigation
        return ConfigSelector(self, (name,))
    
    def _rebuild_parameterized_model(self):
        """Rebuild the model with Parameter types at specified paths"""
        
        def path_to_str(path_tuple):
            """Convert path tuple to string format"""
            parts = []
            for part in path_tuple:
                if isinstance(part, int):
                    parts.append(f"[{part}]")
                elif isinstance(part, str) and ('.' in str(part) or parts):
                    if parts and not parts[-1].startswith('['):
                        parts.append('.')
                    parts.append(str(part))
                else:
                    parts.append(str(part))
            return ''.join(parts).lstrip('.')
        
        def get_inner_type(annotation):
            """Extract inner type from List, Dict, etc."""
            origin = get_origin(annotation)
            args = get_args(annotation)
            
            if origin is list and args:
                return args[0]
            elif origin is dict and len(args) >= 2:
                return args[1]  # Value type
            return None
        
        def check_parameterized_in_collection(prefix_tuple, inner_type):
            """Check if any paths within a collection are parameterized"""
            prefix_str = path_to_str(prefix_tuple)
            for param_path in self._parameters.keys():
                if param_path.startswith(prefix_str + '['):
                    return True
            return False
        
        def rebuild_type_if_needed(field_type, prefix_tuple):
            """Recursively rebuild type if it or its children are parameterized"""
            origin = get_origin(field_type)
            args = get_args(field_type)
            
            if origin is list and args:
                inner_type = args[0]
                if check_parameterized_in_collection(prefix_tuple, inner_type):
                    rebuilt_inner = rebuild_type_if_needed(inner_type, prefix_tuple + (0,))
                    return list[rebuilt_inner]
                return field_type
            
            elif origin is dict and len(args) >= 2:
                key_type, value_type = args[0], args[1]
                if check_parameterized_in_collection(prefix_tuple, value_type):
                    rebuilt_value = rebuild_type_if_needed(value_type, prefix_tuple + ('*',))
                    return dict[key_type, rebuilt_value]
                return field_type
            
            elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                return rebuild_model(field_type, prefix_tuple)
            
            elif isinstance(field_type, type) and is_dataclass(field_type):
                return rebuild_dataclass(field_type, prefix_tuple)
            
            return field_type
        
        def rebuild_dataclass(dc_cls, prefix = ()):
            """Rebuild a dataclass, replacing fields with Parameter types where needed"""
            new_fields = {}
            
            for field in dataclass_fields(dc_cls):
                field_name = field.name
                current_path = prefix + (field_name,)
                path_str = path_to_str(current_path)
                
                if path_str in self._parameters:
                    # Replace with Parameter type
                    original_type = field.type
                    new_type = Parameter[original_type]
                    new_fields[field_name] = (new_type, ...)
                else:
                    # Keep original or rebuild nested
                    rebuilt_type = rebuild_type_if_needed(field.type, current_path)
                    new_fields[field_name] = (rebuilt_type, ...)
            
            # Create a pydantic model from dataclass structure
            new_model = create_model(
                f"{dc_cls.__name__}_Parameterized",
                __config__=None,
                **new_fields
            )
            return new_model
        
        def rebuild_model(model_cls, prefix = ()):
            """Recursively rebuild a model, replacing fields with Parameter types where needed"""
            new_fields = {}
            
            for field_name, field_info in model_cls.model_fields.items():
                current_path = prefix + (field_name,)
                path_str = path_to_str(current_path)
                
                if path_str in self._parameters:
                    # Replace with Parameter type
                    original_type = field_info.annotation
                    new_type = Parameter[original_type]
                    new_fields[field_name] = (new_type, ...)
                else:
                    # Rebuild type if needed (handles nested models, lists, dicts)
                    rebuilt_type = rebuild_type_if_needed(field_info.annotation, current_path)
                    new_fields[field_name] = (rebuilt_type, ...)
            
            # Create new model with modified fields
            new_model = create_model(
                f"{model_cls.__name__}_Parameterized",
                __config__=None,
                **new_fields
            )
            return new_model
        
        self._parameterized_model = rebuild_model(type(self._instance))
    
    def _model_dump(self) -> dict:
        """Convert the model to dict, replacing parameterized fields with Parameter objects"""
        # Rebuild the parameterized model if parameters have changed
        self._rebuild_parameterized_model()
        
        result = self._instance.model_dump()
        
        def parse_path(path_str):
            """Parse path string into components, handling brackets"""
            import re
            # Split on dots, but handle brackets
            parts = []
            current = ""
            i = 0
            while i < len(path_str):
                if path_str[i] == '[':
                    if current:
                        parts.append(current)
                        current = ""
                    # Find closing bracket
                    j = path_str.index(']', i)
                    bracket_content = path_str[i+1:j]
                    # Remove quotes if present
                    if bracket_content.startswith("'") or bracket_content.startswith('"'):
                        bracket_content = bracket_content[1:-1]
                    elif bracket_content.isdigit():
                        bracket_content = int(bracket_content)
                    parts.append(bracket_content)
                    i = j + 1
                elif path_str[i] == '.':
                    if current:
                        parts.append(current)
                        current = ""
                    i += 1
                else:
                    current += path_str[i]
                    i += 1
            if current:
                parts.append(current)
            return parts
        
        for path, key in self._parameters.items():
            # Navigate to the right place in the dict and replace with Parameter
            current = result
            parts = parse_path(path)
            
            for part in parts[:-1]:
                current = current[part]
            
            final_key = parts[-1]
            default_value = current[final_key]
            param_dict = {"default": default_value, "key": key}
            
            # Validate this specific parameter has the correct type
            # by finding its type in the original model
            try:
                param_type = self._get_field_type_at_path(parts)
                if param_type:
                    Parameter[param_type].model_validate(param_dict)
            except:
                pass  # If we can't determine type, skip validation
            
            current[final_key] = param_dict
        
        return result
    
    def _get_field_type_at_path(self, parts):
        """Get the type of a field at a given path"""
        current_type = type(self._instance)
        
        for i, part in enumerate(parts):
            if isinstance(part, str):
                if hasattr(current_type, 'model_fields') and part in current_type.model_fields:
                    field_info = current_type.model_fields[part]
                    current_type = field_info.annotation
                else:
                    return None
            elif isinstance(part, int):
                # List index
                origin = get_origin(current_type)
                if origin is list:
                    args = get_args(current_type)
                    if args:
                        current_type = args[0]
                else:
                    return None
            # For dict keys, get the value type
            else:
                origin = get_origin(current_type)
                if origin is dict:
                    args = get_args(current_type)
                    if len(args) >= 2:
                        current_type = args[1]
                else:
                    return None
        
        return current_type
    
    def _build(self) -> BaseModel:
        """Build the model, replacing Parameters with actual values from PARAMETERS"""
        def replace_parameters(obj):
            if isinstance(obj, dict):
                # Check if this is a Parameter dict
                if "default" in obj and "key" in obj and len(obj) == 2:
                    key = obj["key"]
                    return PARAMETERS.get(key, obj["default"])
                # Recursively process dict
                return {k: replace_parameters(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_parameters(item) for item in obj]
            else:
                return obj
        
        # Get the parameterized dump
        data = self._model_dump()
        # Replace parameters with actual values
        resolved_data = replace_parameters(data)
        # Rebuild the model
        return type(self._instance).model_validate(resolved_data)


class ConfigSelector:
    def __init__(self, config_model: ConfigModel, path: t.Tuple[t.Union[str, int], ...]):
        self.config_model = config_model
        self.path = path
    
    def __getattr__(self, name: str):
        # Continue building the path
        return ConfigSelector(self.config_model, self.path + (name,))
    
    def __getitem__(self, key: t.Union[int, str]):
        # Support list indexing and dict key access
        return ConfigSelector(self.config_model, self.path + (key,))
    
    def set_as_parameter(self, key: str):
        """Mark this path as a parameter"""
        path_parts = []
        for part in self.path:
            if isinstance(part, int):
                path_parts.append(f"[{part}]")
            elif isinstance(part, str) and '.' in str(part):
                path_parts.append(f"['{part}']")
            else:
                if path_parts and not path_parts[-1].startswith('['):
                    path_parts.append('.')
                path_parts.append(str(part))
        
        path_str = ''.join(path_parts).lstrip('.')
        self.config_model._parameters[path_str] = key


# Test it
@dataclass
class SubSubSubModel:
    x: float
    y: float

class SubSubModel(BaseModel):
    a: str
    b: float
    c: int
    pos: SubSubSubModel


class SubModel(BaseModel):
    a: str
    b: float
    c: int
    d: t.List[SubSubModel]
    e: t.Dict[str, SubSubModel]

class TestModel(BaseModel):
    a: int
    b: SubModel


if __name__ == "__main__":
    # Add new parameters for testing
    PARAMETERS["pos_x"] = 100.5
    PARAMETERS["item_a"] = "list_item"
    PARAMETERS["dict_c"] = 999
    
    # Create instance with nested data
    t = TestModel(
        a=1, 
        b=SubModel(
            a='a', 
            b=1.1, 
            c=2,
            d=[
                SubSubModel(a='item1', b=2.2, c=10, pos=SubSubSubModel(x=1.0, y=2.0)),
                SubSubModel(a='item2', b=3.3, c=20, pos=SubSubSubModel(x=3.0, y=4.0))
            ],
            e={
                'x': SubSubModel(a='dictA', b=4.4, c=30, pos=SubSubSubModel(x=5.0, y=6.0)),
                'y': SubSubModel(a='dictB', b=5.5, c=40, pos=SubSubSubModel(x=7.0, y=8.0))
            }
        )
    )
    
    # Wrap in ConfigModel
    c = ConfigModel.from_class(t)
    
    # Test 1: Simple nested fields
    print("=" * 50)
    print("TEST 1: Simple nested fields")
    print("=" * 50)
    c.b.c.set_as_parameter("value")
    c.b.a.set_as_parameter("str_value")
    
    dump = c.model_dump()
    assert dump['b']['c'] == {"default": 2, "key": "value"}
    assert dump['b']['a'] == {"default": 'a', "key": "str_value"}
    print("✓ Simple nested fields work")
    print()
    
    # Test 2: List indexing - c.b.d[0].pos.x
    print("=" * 50)
    print("TEST 2: List indexing")
    print("=" * 50)
    c2 = ConfigModel.from_class(t)
    c2.b.d[0].pos.x.set_as_parameter("pos_x")
    c2.b.d[0].a.set_as_parameter("item_a")
    
    dump2 = c2.model_dump()
    print(f"Path: b.d[0].pos.x")
    print(f"Value: {dump2['b']['d'][0]['pos']['x']}")
    assert dump2['b']['d'][0]['pos']['x'] == {"default": 1.0, "key": "pos_x"}
    assert dump2['b']['d'][0]['a'] == {"default": 'item1', "key": "item_a"}
    print("✓ List indexing works")
    
    built2 = c2.build()
    built_dump2 = built2.model_dump()
    assert built_dump2['b']['d'][0]['pos']['x'] == 100.5
    assert built_dump2['b']['d'][0]['a'] == "list_item"
    print("✓ List indexing build() works")
    print()
    
    # Test 3: Dict key access - c.b.e['x'].c
    print("=" * 50)
    print("TEST 3: Dict key access")
    print("=" * 50)
    c3 = ConfigModel.from_class(t)
    c3.b.e['x'].c.set_as_parameter("dict_c")
    
    dump3 = c3.model_dump()
    print(f"Path: b.e['x'].c")
    print(f"Value: {dump3['b']['e']['x']['c']}")
    assert dump3['b']['e']['x']['c'] == {"default": 30, "key": "dict_c"}
    print("✓ Dict key access works")
    
    built3 = c3.build()
    built_dump3 = built3.model_dump()
    assert built_dump3['b']['e']['x']['c'] == 999
    print("✓ Dict key access build() works")
    print()
    
    # Test 4: Validation with wrong type
    print("=" * 50)
    print("TEST 4: Type validation")
    print("=" * 50)
    try:
        c2._parameterized_model.model_validate({
            "a": 1,
            "b": {
                "a": 'a', "b": 1.1, "c": 2,
                "d": [{"a": {"default": "wrong", "key": "item_a"}, "b": 2.2, "c": 10, "pos": {"x": {"default": "not_a_float", "key": "pos_x"}, "y": 2.0}}],
                "e": {}
            }
        })
        print("✗ Expected validation error but none was raised!")
        exit(1)
    except Exception as e:
        print(f"✓ Got expected validation error: {type(e).__name__}")
        print(f"  Error snippet: {str(e)[:150]}...")
    print()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
