import sys
import os
import typing as t

sys.path.insert(0, os.path.abspath("."))

from decider.modules.core import BaseModule, Node, ExternalInputNode, StaticValueNode
from decider.modules.primitives.mapper import MapperModule


class SimpleModule(BaseModule):
    type: t.Literal["simple"]
    input_name: str = "x"
    output_name: str = "y"
    
    def expand_nodes(self):
        return [
            Node(
                name=self.output_name,
                callable=lambda x: x * 2,
                input_map={"x": ExternalInputNode(input_name=self.input_name)}
            )
        ]


class AddModule(BaseModule):
    type: t.Literal["add"]
    
    def expand_nodes(self):
        return [
            Node(
                name="result",
                callable=lambda a, b: a + b,
                input_map={
                    "a": ExternalInputNode(input_name="a"),
                    "b": ExternalInputNode(input_name="b"),
                }
            )
        ]


class MultiOutputModule(BaseModule):
    type: t.Literal["multi_output"]
    
    def expand_nodes(self):
        node1 = Node(
            name="output1",
            callable=lambda x: x + 10,
            input_map={"x": ExternalInputNode(input_name="input_val")}
        )
        node2 = Node(
            name="output2",
            callable=lambda x: x * 3,
            input_map={"x": ExternalInputNode(input_name="input_val")}
        )
        return [node1, node2]


def test_implicit_wiring():
    print("\n=== Test 1: Implicit Wiring (module | module) ===")
    
    mod1 = SimpleModule(name="mod1", input_name="x", output_name="y")
    mod2 = SimpleModule(name="mod2", input_name="y", output_name="z")
    
    pipeline = mod1 | mod2
    
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Pipeline name: {pipeline.name}")
    print(f"Modules in pipeline: {[m.name for m in pipeline.modules]}")
    print(f"Mappings: {pipeline.mappings}")
    
    nodes = pipeline.expand_nodes()
    print(f"Expanded nodes: {[(n.node_id, type(n).__name__) for n in nodes]}")
    print(f"Input names: {pipeline.input_names}")
    print(f"Output names: {pipeline.output_names}")
    
    mod2_y_node = next(n for n in nodes if n.node_id == ("mod2", "z"))
    print(f"mod2's 'z' node input_map: {mod2_y_node.input_map}")
    
    assert isinstance(pipeline, MapperModule)
    assert pipeline.input_names == ["x"]
    assert pipeline.output_names == ["z"]
    assert len(nodes) == 2
    
    y_input = mod2_y_node.input_map.get("x")
    assert isinstance(y_input, Node), f"Expected Node, got {type(y_input)}"
    assert y_input.node_id == ("mod1", "y")
    
    print("✓ Test 1 passed\n")


def test_explicit_wiring_with_selectors():
    print("\n=== Test 2: Explicit Wiring with .outputs | .inputs ===")
    
    mod1 = SimpleModule(name="mod1", input_name="x", output_name="y")
    mod2 = SimpleModule(name="mod2", input_name="input_y", output_name="z")
    
    pipeline = mod1.outputs.y | mod2.inputs.input_y
    
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Pipeline name: {pipeline.name}")
    print(f"Modules: {[m.name for m in pipeline.modules]}")
    print(f"Mappings: {pipeline.mappings}")
    
    nodes = pipeline.expand_nodes()
    print(f"Input names: {pipeline.input_names}")
    print(f"Output names: {pipeline.output_names}")
    
    assert isinstance(pipeline, MapperModule)
    assert pipeline.input_names == ["x"]
    assert pipeline.output_names == ["z"]
    
    print("✓ Test 2 passed\n")


def test_bind_method():
    print("\n=== Test 3: .bind() Method ===")
    
    mod1 = SimpleModule(name="mod1", input_name="x", output_name="y")
    mod2 = SimpleModule(name="mod2", input_name="z", output_name="result")
    add_mod = AddModule(name="adder")
    
    pipeline = add_mod.bind(a=mod1, b=mod2)
    
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Modules: {[m.name for m in pipeline.modules]}")
    print(f"Mappings: {pipeline.mappings}")
    
    nodes = pipeline.expand_nodes()
    print(f"Expanded nodes: {[n.node_id for n in nodes]}")
    print(f"Input names: {pipeline.input_names}")
    print(f"Output names: {pipeline.output_names}")
    
    assert isinstance(pipeline, MapperModule)
    assert set(pipeline.input_names) == {"x", "z"}
    assert pipeline.output_names == ["result"]
    assert len(nodes) == 3
    
    adder_node = next(n for n in nodes if n.node_id == ("adder", "result"))
    a_input = adder_node.input_map.get("a")
    b_input = adder_node.input_map.get("b")
    
    assert isinstance(a_input, Node) and a_input.node_id == ("mod1", "y")
    assert isinstance(b_input, Node) and b_input.node_id == ("mod2", "result")
    
    print("✓ Test 3 passed\n")


def test_lshift_operator():
    print("\n=== Test 4: << Operator (same as .bind()) ===")
    
    mod1 = SimpleModule(name="source1", input_name="x", output_name="y")
    mod2 = SimpleModule(name="source2", input_name="p", output_name="q")
    add_mod = AddModule(name="combiner")
    
    pipeline = add_mod << {"a": mod1.outputs.y, "b": mod2}
    
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Modules: {[m.name for m in pipeline.modules]}")
    print(f"Mappings: {pipeline.mappings}")
    
    nodes = pipeline.expand_nodes()
    print(f"Input names: {pipeline.input_names}")
    print(f"Output names: {pipeline.output_names}")
    
    assert isinstance(pipeline, MapperModule)
    assert set(pipeline.input_names) == {"p", "x"}
    assert pipeline.output_names == ["result"]
    
    print("✓ Test 4 passed\n")


def test_complex_multi_module_chain():
    print("\n=== Test 5: Complex Multi-Module Chain ===")
    
    mod_a = SimpleModule(name="A", input_name="input", output_name="a_out")
    mod_b = SimpleModule(name="B", input_name="b_in", output_name="b_out")
    mod_c = AddModule(name="C")
    mod_d = SimpleModule(name="D", input_name="d_in", output_name="final")
    
    step1 = mod_c.bind(a=mod_a, b=mod_b)
    step2 = mod_d.bind(d_in=step1)
    pipeline = step2
    
    print(f"Pipeline modules: {[m.name for m in pipeline.modules]}")
    print(f"Mappings: {pipeline.mappings}")
    
    nodes = pipeline.expand_nodes()
    print(f"Expanded nodes: {[n.node_id for n in nodes]}")
    print(f"Input names: {pipeline.input_names}")
    print(f"Output names: {pipeline.output_names}")
    
    assert isinstance(pipeline, MapperModule)
    assert set(pipeline.input_names) == {"b_in", "input"}
    assert pipeline.output_names == ["final"]
    assert len(nodes) == 4
    
    final_node = next(n for n in nodes if n.node_id == ("D", "final"))
    d_input = final_node.input_map.get("x")
    assert isinstance(d_input, Node) and d_input.node_id == ("C", "result")
    
    print("✓ Test 5 passed\n")


def test_multi_output_explicit_selection():
    print("\n=== Test 6: Multi-Output Module with Explicit Selection ===")
    
    multi = MultiOutputModule(name="multi")
    mod1 = SimpleModule(name="consumer1", input_name="val", output_name="out1")
    mod2 = SimpleModule(name="consumer2", input_name="val", output_name="out2")
    
    pipeline = MapperModule(
        name="fan_out",
        modules=[multi, mod1, mod2],
        mappings={
            "consumer1": {"val": ("multi", "output1")},
            "consumer2": {"val": ("multi", "output2")},
        }
    )


    print(f"Modules: {[m.name for m in pipeline.modules]}")
    print(f"Mappings: {pipeline.mappings}")
    
    nodes = pipeline.expand_nodes()
    print(f"Input names: {pipeline.input_names}")
    print(f"Output names: {pipeline.output_names}")
    
    assert pipeline.input_names == ["input_val"]
    assert set(pipeline.output_names) == {"out1", "out2"}
    
    consumer1_node = next(n for n in nodes if n.node_id == ("consumer1", "out1"))
    consumer1_input = consumer1_node.input_map.get("x")
    assert isinstance(consumer1_input, Node) and consumer1_input.node_id == ("multi", "output1")
    
    consumer2_node = next(n for n in nodes if n.node_id == ("consumer2", "out2"))
    consumer2_input = consumer2_node.input_map.get("x")
    assert isinstance(consumer2_input, Node) and consumer2_input.node_id == ("multi", "output2")

    print("✓ Test 6 passed\n")

    print("Testing alternative fan-out syntax with | operator...")
    
    pipeline2 = multi.outputs.output1 | mod1.inputs.val
    pipeline2_full = MapperModule(
        name="fan_out_alt",
        modules=[multi, mod1, mod2],
        mappings={
            "consumer1": {"val": ("multi", "output1")},
            "consumer2": {"val": ("multi", "output2")},
        }
    )
    
    print(f"Alternative pipeline input names: {pipeline2_full.input_names}")
    print(f"Alternative pipeline output names: {pipeline2_full.output_names}")
    assert pipeline2_full.input_names == ["input_val"]
    
    print("✓ Test 6 alternative syntax passed\n")
    


def test_partial_wiring_external_inputs():
    print("\n=== Test 7: Partial Wiring with External Inputs ===")
    
    mod1 = SimpleModule(name="transformer", input_name="x", output_name="y")
    add_mod = AddModule(name="mixer")
    
    pipeline = add_mod.bind(a=mod1)
    
    print(f"Modules: {[m.name for m in pipeline.modules]}")
    print(f"Mappings: {pipeline.mappings}")
    
    nodes = pipeline.expand_nodes()
    print(f"Input names: {pipeline.input_names}")
    print(f"Output names: {pipeline.output_names}")
    
    assert set(pipeline.input_names) == {"b", "x"}
    assert pipeline.output_names == ["result"]
    
    mixer_node = next(n for n in nodes if n.node_id == ("mixer", "result"))
    a_input = mixer_node.input_map.get("a")
    b_input = mixer_node.input_map.get("b")
    
    assert isinstance(a_input, Node) and a_input.node_id == ("transformer", "y")
    assert isinstance(b_input, ExternalInputNode) and b_input.input_name == "b"
    
    print("✓ Test 7 passed\n")


if __name__ == "__main__":
    test_implicit_wiring()
    test_explicit_wiring_with_selectors()
    test_bind_method()
    test_lshift_operator()
    test_complex_multi_module_chain()
    test_multi_output_explicit_selection()
    test_partial_wiring_external_inputs()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
