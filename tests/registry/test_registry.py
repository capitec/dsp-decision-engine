from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from decider.modules.core import BaseModule
from decider.registry import BaseRegistryModule, import_path, ref


class Step(ABC):
    __slots__ = ()

    @abstractmethod
    def run(self) -> str: ...


class Root(Step, BaseRegistryModule, root=True):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str

    def run(self) -> str:
        return self.name


RootRef = ref(Root)


class Plain(Root):
    x: int = 1


class Aliased(Root):
    type: t.Literal["aliased"] = "aliased"
    y: int = 2


class Sequence(Root):
    type: t.Literal["sequence"] = "sequence"
    steps: list[RootRef] = []
    by_name: dict[str, RootRef] = {}


class Holder(BaseModel):
    step: RootRef


def test_class_without_alias_is_addressable_by_import_path():
    assert Root.resolve(import_path(Plain)) is Plain
    assert import_path(Plain) == f"{__name__}:Plain"


def test_type_defaults_to_import_path_and_dumps_it():
    p = Plain(name="p")
    assert p.type == import_path(Plain)
    assert p.model_dump() == {"type": import_path(Plain), "name": "p", "x": 1}


def test_wrong_type_is_rejected_on_load():
    with pytest.raises(ValidationError, match="does not match"):
        Plain.model_validate({"type": "aliased", "name": "p"})


def test_alias_loads_by_either_spelling_and_dumps_as_alias():
    assert Root.resolve("aliased") is Aliased
    assert Root.resolve(import_path(Aliased)) is Aliased
    by_path = Aliased.model_validate({"type": import_path(Aliased), "name": "a"})
    assert by_path.model_dump()["type"] == "aliased"
    via_ref = TypeAdapter(RootRef).validate_python({"type": import_path(Aliased), "name": "a"})
    assert type(via_ref) is Aliased and via_ref.type == "aliased"


def test_import_path_imports_the_module(tmp_path, monkeypatch):
    (tmp_path / "registry_plugin_mod.py").write_text(
        "from decider.modules.core import BaseModule\n"
        "class Plugin(BaseModule):\n"
        "    x: int = 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cls = BaseModule.resolve("registry_plugin_mod:Plugin")
    assert cls.__name__ == "Plugin" and issubclass(cls, BaseModule)


class NotAStep(BaseModel):
    type: str = "x"


@pytest.mark.parametrize(
    "tag", ["os:system", "decimal:Decimal", f"{__name__}:NotAStep", "no_such_module_xyz:Thing"]
)
def test_import_path_to_a_non_registered_class_is_rejected(tag):
    with pytest.raises(LookupError, match="not a registered"):
        Root.resolve(tag)
    with pytest.raises(ValidationError, match="not a registered"):
        Holder.model_validate({"step": {"type": tag, "name": "n"}})


def test_independent_roots_never_share_entries():
    with pytest.raises(LookupError):
        BaseModule.resolve(import_path(Plain))
    assert "aliased" not in BaseModule.registered_extensions()


def test_two_classes_claiming_one_alias_is_an_error_naming_both():
    class First(Root):
        type: t.Literal["taken_once"] = "taken_once"

    with pytest.raises(ValueError, match="taken_once") as err:

        class Second(Root):
            type: t.Literal["taken_once"] = "taken_once"

    assert import_path(First) in str(err.value)
    assert "Second" in str(err.value)


def test_redefining_the_same_qualified_name_replaces_the_entry():
    class Reloaded(Root):
        type: t.Literal["reloaded"] = "reloaded"
        v: int = 1

    old = Reloaded

    class Reloaded(Root):  # noqa: F811
        type: t.Literal["reloaded"] = "reloaded"
        v: int = 2

    assert Reloaded is not old
    assert Root.resolve("reloaded") is Reloaded
    assert Root.resolve(import_path(Reloaded)) is Reloaded
    assert TypeAdapter(RootRef).validate_python({"type": "reloaded", "name": "r"}).v == 2


def test_distinct_classes_coexist():
    class ModA(Root):
        type: t.Literal["mod_a"] = "mod_a"

    class ModB(Root):
        type: t.Literal["mod_b"] = "mod_b"

    adapter = TypeAdapter(RootRef)
    assert type(adapter.validate_python({"type": "mod_a", "name": "a"})) is ModA
    assert type(adapter.validate_python({"type": "mod_b", "name": "b"})) is ModB


def test_subclass_inheriting_an_alias_must_declare_its_own_type():
    with pytest.raises(ValueError, match="claimed by both"):

        class Sub(Aliased):
            pass

    class SubByPath(Aliased):
        type: str

    assert SubByPath(name="s").type == import_path(SubByPath)


def test_abstract_subclasses_are_not_registered():
    class Family(Root):
        @abstractmethod
        def extra(self) -> int: ...

    assert import_path(Family) not in Root.registered_extensions()

    class AliasedFamily(Root):
        type: t.Literal["family_member"] = "family_member"

        @abstractmethod
        def extra(self) -> int: ...

    class Member(AliasedFamily):
        def extra(self) -> int:
            return 1

    assert Root.resolve("family_member") is Member


def test_unknown_tag_suggests_the_closest_registered_one():
    with pytest.raises(LookupError, match="Did you mean 'sequence'"):
        Root.resolve("seqeunce")
    with pytest.raises(ValidationError, match="Did you mean 'aliased'"):
        Holder.model_validate({"step": {"type": "alaised", "name": "n"}})


def test_resolved_class_validates_its_fields_normally():
    with pytest.raises(ValidationError) as err:
        Holder.model_validate({"step": {"type": "aliased", "name": "n", "y": "bad"}})
    assert err.value.errors()[0]["loc"] == ("step", "y")
    with pytest.raises(ValidationError, match="extra"):
        Holder.model_validate({"step": {"type": "aliased", "name": "n", "nope": 1}})


def test_ref_passes_instances_through():
    a = Aliased(name="a")
    assert Holder(step=a).step is a


def test_nested_subclass_fields_survive_a_json_round_trip():
    held = Holder(step=Aliased(name="a", y=7))
    assert held.model_dump()["step"] == {"type": "aliased", "name": "a", "y": 7}
    back = Holder.model_validate_json(held.model_dump_json())
    assert type(back.step) is Aliased and back.step.y == 7


def test_list_dict_and_recursive_refs_round_trip():
    doc = {
        "type": "sequence",
        "name": "outer",
        "steps": [
            {"type": import_path(Plain), "name": "p", "x": 5},
            {"type": "sequence", "name": "inner", "steps": [{"type": "aliased", "name": "a", "y": 9}]},
        ],
        "by_name": {"k": {"type": "aliased", "name": "b"}},
    }
    seq = TypeAdapter(RootRef).validate_python(doc)
    assert type(seq.steps[1].steps[0]) is Aliased
    assert seq.by_name["k"].run() == "b"
    back = Sequence.model_validate_json(seq.model_dump_json())
    assert back == seq
    assert back.model_dump()["steps"][1]["steps"][0] == {"type": "aliased", "name": "a", "y": 9}


def test_nested_errors_keep_their_location():
    with pytest.raises(ValidationError) as err:
        Sequence.model_validate(
            {"name": "s", "steps": [{"type": "sequence", "name": "n", "steps": [{"type": "aliased", "name": "a", "y": "x"}]}]}
        )
    assert err.value.errors()[0]["loc"] == ("steps", 0, "steps", 0, "y")


def test_json_schema_covers_every_registered_class():
    schemas = Root.json_schema()
    assert {"aliased", "sequence", import_path(Plain)} <= schemas.keys()
    assert schemas["aliased"]["properties"]["type"]["const"] == "aliased"
    assert "y" in schemas["aliased"]["properties"]
    assert schemas[import_path(Plain)]["properties"]["type"]["const"] == import_path(Plain)


def test_root_composes_with_a_slotted_abc():
    p = Plain(name="p")
    assert isinstance(p, Step) and p.run() == "p"
    with pytest.raises(ValidationError):
        p.name = "changed"
    with pytest.raises(TypeError):

        class Incomplete(Step, BaseRegistryModule, root=True):
            pass

        Incomplete(type="t")
