"""
Output schema definitions and validation for tree execution results.

This module contains all the schema-related classes, types, and validation logic
for the new output system (nodeOutputFormatVersion=1).
"""

import typing as t
import typing_extensions as t_ext
from enum import Enum
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    model_validator,
    field_validator,
    Discriminator,
    RootModel,
    Tag,
)


class FieldType(str, Enum):
    """Supported field types for output schema"""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CUSTOM = "custom"
    LIST = "list"


class CustomTypeKind(str, Enum):
    """Supported custom type kinds"""

    ENUM = "enum"
    RECORD = "record"


class RecordDefinition(BaseModel):
    """Record definition containing fields and records"""

    fields: t.Dict[
        str, t.Literal[FieldType.STRING, FieldType.NUMBER, FieldType.BOOLEAN]
    ]
    records: t.List[t.Dict[str, t.Any]] = Field(default_factory=list)


class EnumDefinition(BaseModel):
    """Enum definition containing values"""

    values: t.Set[str]


class _RecordCustomTypeInfo(BaseModel):
    """Snapshot of a custom record type definition at time of use"""

    id: str
    name: str
    type_kind: t.Literal[CustomTypeKind.RECORD] = CustomTypeKind.RECORD
    definition: RecordDefinition
    display_format: t.Optional[str] = None
    _parsed_records: t.Optional[t.List[t.Dict[str, t.Any]]] = PrivateAttr(default=None)

    @property
    def parsed_records(self) -> t.List[t.Dict[str, t.Any]]:
        """Get parsed records with validated and converted field values"""
        if self._parsed_records is None:
            parsed_data = [{} for _ in range(len(self.definition.records))]
            for record in self.definition.records:
                record_id = record.get("id")
                if record_id is None:
                    raise ValueError("Record missing 'id' field")
                record_id = int(record_id)
                if record_id >= len(parsed_data):
                    parsed_data.extend(
                        [{} for _ in range(record_id - len(parsed_data) + 1)]
                    )
                if parsed_data[record_id]:
                    raise ValueError(
                        f"Duplicate record id {record_id} in custom record type"
                    )
                parsed_record = {}
                for field_name, field_type in self.definition.fields.items():
                    if field_name not in record:
                        parsed_record[field_name] = None
                    else:
                        field_value = record[field_name]
                        parsed_field = _PrimitiveFieldTypes(
                            field_type=field_type, value=field_value
                        ).root
                        parsed_record[field_name] = parsed_field.parsed_value
                parsed_data[record_id] = parsed_record
            self._parsed_records = parsed_data
        return self._parsed_records


class _EnumCustomTypeInfo(BaseModel):
    """Snapshot of a custom enum type definition at time of use"""

    id: str
    name: str
    type_kind: t.Literal[CustomTypeKind.ENUM] = CustomTypeKind.ENUM
    definition: EnumDefinition


_CustomTypeInfo = t.Annotated[
    t.Union[_RecordCustomTypeInfo, _EnumCustomTypeInfo],
    Field(discriminator="type_kind"),
]


class OutputField(BaseModel):
    """Output field definition for the schema"""

    id: str
    field_name: str
    field_type: FieldType
    list_type: t.Optional[
        t.Literal[
            FieldType.STRING, FieldType.NUMBER, FieldType.BOOLEAN, FieldType.CUSTOM
        ]
    ] = None
    is_required: bool = True
    custom_type: t.Optional[_CustomTypeInfo] = None  # Embedded snapshot
    custom_type_id: t.Optional[str] = None  # Reference to custom type

    @model_validator(mode="after")
    def validate_type_fields(self) -> "t_ext.Self":
        if self.field_type == FieldType.CUSTOM:
            if self.custom_type is None:
                raise ValueError("Custom type must be provided for CUSTOM field type")
        if self.field_type == FieldType.LIST:
            if self.list_type is None:
                raise ValueError("List type must be provided for LIST field type")
            if self.list_type == FieldType.CUSTOM:
                assert (
                    self.custom_type is not None
                ), "Custom type must be provided for CUSTOM list type"
        return self


# Private field validation types
_T = t.TypeVar("T")
_V = t.TypeVar("V", bound=FieldType)


class _BaseFieldValueType(BaseModel, t.Generic[_V, _T]):
    field_type: _V
    value: _T

    @property
    def parsed_value(self):
        return self.value


class _StringFieldValue(_BaseFieldValueType[t.Literal[FieldType.STRING], str]):
    @field_validator("value", mode="before")
    @classmethod
    def convert_to_string(cls, v):
        """Convert input values to string"""
        if v is None:
            return None
        return str(v)


class _NumberFieldValue(
    _BaseFieldValueType[t.Literal[FieldType.NUMBER], t.Union[int, float]]
):
    pass


class _BooleanFieldValue(_BaseFieldValueType[t.Literal[FieldType.BOOLEAN], bool]):
    @field_validator("value", mode="before")
    @classmethod
    def convert_to_bool(cls, v):
        """Convert input values to boolean"""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        if isinstance(v, (int, float)):
            return bool(v)
        return bool(v)


_TPrimitiveFieldTypes = t.Annotated[
    t.Union[_StringFieldValue, _NumberFieldValue, _BooleanFieldValue],
    Discriminator("field_type"),
]


class _PrimitiveFieldTypes(RootModel):
    root: _TPrimitiveFieldTypes


class _CustomEnumFieldType(_BaseFieldValueType[t.Literal[FieldType.CUSTOM], str]):
    custom_type: _EnumCustomTypeInfo

    @model_validator(mode="after")
    def validate_values(self):
        assert (
            self.value in self.custom_type.definition.values
        ), f"Must be one of {self.custom_type.definition.values}, got '{self.value}'"
        return self


class _CustomRecordFieldType(_BaseFieldValueType[t.Literal[FieldType.CUSTOM], int]):
    custom_type: _RecordCustomTypeInfo
    _parsed_values: t.Dict[str, t.Union[str, int, float]] = PrivateAttr()

    @model_validator(mode="after")
    def validate_values(self):
        assert (
            0 <= self.value < len(self.custom_type.parsed_records)
        ), f"Record id must be between 0 and {len(self.custom_type.parsed_records)-1}, got '{self.value}'"
        self._parsed_values = self.custom_type.parsed_records[self.value]
        return self

    @property
    def parsed_value(self):
        return self._parsed_values


def _discriminate_custom_type(v):
    if isinstance(v, dict):
        # Raw dict during parsing
        custom_type = v.get("custom_type", {})
    else:
        # Already parsed object
        custom_type = getattr(v, "custom_type", {})
    if isinstance(custom_type, dict):
        return custom_type.get("type_kind")
    return getattr(custom_type, "type_kind", None)


_TCustomTypes = t.Annotated[
    t.Union[
        t.Annotated[_CustomEnumFieldType, Tag("enum")],
        t.Annotated[_CustomRecordFieldType, Tag("record")],
    ],
    Discriminator(_discriminate_custom_type),
]

_TListFieldTypes = t.Annotated[
    t.Union[_StringFieldValue, _NumberFieldValue, _BooleanFieldValue, _TCustomTypes],
    Discriminator("field_type"),
]


class _ListFieldTypes(RootModel):
    root: _TListFieldTypes


class _ListFieldType(_BaseFieldValueType[t.Literal[FieldType.LIST], list]):
    list_type: t.Literal[
        FieldType.STRING, FieldType.BOOLEAN, FieldType.NUMBER, FieldType.CUSTOM
    ]
    custom_type: t.Optional[_CustomTypeInfo] = None
    _parsed_values: t.List[t.Union[str, int, float]] = PrivateAttr()

    @model_validator(mode="after")
    def validate_values(self):
        type_kwargs = {"field_type": self.list_type}
        if self.list_type == FieldType.CUSTOM:
            assert (
                self.custom_type is not None
            ), "Expected custom type to be an enum type if field type is CUSTOM"
            type_kwargs["custom_type"] = self.custom_type
        self._parsed_values = [
            _ListFieldTypes(value=v, **type_kwargs).root.parsed_value
            for v in self.value
        ]
        return self

    @property
    def parsed_value(self):
        return self._parsed_values


_TFieldValueTypes = t.Annotated[
    t.Union[
        _StringFieldValue,
        _NumberFieldValue,
        _BooleanFieldValue,
        _TCustomTypes,
        _ListFieldType,
    ],
    Discriminator("field_type"),
]


class _FieldValueTypes(RootModel):
    root: _TFieldValueTypes


class OutputSchema(BaseModel):
    """Schema definition for structured output"""

    fields: t.List[OutputField] = Field(default_factory=list)
    display_format: t.Optional[str] = None  # Template for node display
    default_values: t.Optional[t.Dict[str, t.Any]] = None

    def has_default_values(self) -> bool:
        if self.default_values is None:
            return False
        has_required_fields = any(field.is_required for field in self.fields)
        # Unfortunately the ui still puts this as an empty dict even if no default values are set
        # So for now we work around it.
        return len(self.default_values) > 0 and has_required_fields

    @model_validator(mode="after")
    def validate_default_values(self) -> "t_ext.Self":
        if self.has_default_values():
            errors = self.validate_data(self.default_values)
            if errors:
                raise ValueError(f"Default values validation errors: {errors}")
        return self

    def validate_data(self, data: t.Dict[str, t.Any]) -> t.List[str]:
        """Validate output data against schema. Returns list of validation errors."""
        errors = []

        # Check for required fields
        for field in self.fields:
            if field.field_name not in data:
                if field.is_required:
                    errors.append(f"Field '{field.field_name}' is required but missing")
                continue
            try:
                parsed_field = _FieldValueTypes(
                    **field.model_dump(), value=data[field.field_name]
                )
            except ValueError as e:
                errors.append(str(e))
                continue
            # The parsed field will automatically convert like "5" -> 5
            data[field.field_name] = parsed_field.root.parsed_value
        return errors

    def columns(self):
        return [field.field_name for field in self.fields]

    def dtypes(self):
        dtype_mapping = {}
        for field in self.fields:
            if field.field_type == FieldType.STRING:
                dtype_mapping[field.field_name] = "object"
            elif field.field_type == FieldType.NUMBER:
                dtype_mapping[field.field_name] = "float64"
            elif field.field_type == FieldType.BOOLEAN:
                dtype_mapping[field.field_name] = "boolean"
            elif field.field_type == FieldType.LIST:
                # Lists are stored as objects in pandas
                dtype_mapping[field.field_name] = "object"
            elif field.field_type == FieldType.CUSTOM:
                # Custom types (enums, records) are stored as objects
                dtype_mapping[field.field_name] = "object"
            else:
                dtype_mapping[field.field_name] = "object"
        return dtype_mapping
