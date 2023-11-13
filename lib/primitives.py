
from hamilton.function_modifiers import base
import typing
import pandas as pd
import abc
import os
import sys
from hamilton import node as ham_node

# from hamilton.function_modifiers import inject, value, source, recursive, dependencies
from hamilton import function_modifiers as fm
# from hamilton.function_modifiers.recursive import NON_FINAL_TAGS

import importlib
from lib import creators, builtin_funcs
from lib.builtin_funcs import base_ops
from .helpers import OverrideNodeExpander, infer_inject_parameter
import functools

class Primitive(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def expand_nodes(
        cls, 
        prim_config: typing.Dict[str, typing.Any], 
        function_map: typing.Dict[str, typing.Callable], 
        config: typing.Dict[str, typing.Any], 
        **kwargs: typing.Dict[str, typing.Any]
    ) -> typing.Collection[ham_node.Node]:
        raise NotImplemented()

class KwPrimitive(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def expand_nodes(
        cls, 
        prim_key:str, 
        prim_config: typing.Dict[str, typing.Any], 
        function_map: typing.Dict[str, typing.Callable], 
        config: typing.Dict[str, typing.Any], 
        **kwargs: typing.Dict[str, typing.Any]
    ) -> typing.Collection[ham_node.Node]:
        raise NotImplemented()


class Step(KwPrimitive):
    @staticmethod
    def expand_nodes(
            step_name: str, 
            step_config: typing.Dict[str, typing.Any], 
            function_map: typing.Dict[str, typing.Callable], 
            config: dict,
            **kwargs
    ):
        
        fn = function_map[step_config["step_type"]]
        additional_conf = {}
        if "inputs" in step_config:
            injector = fm.inject(
                **{
                    input_key: infer_inject_parameter(input_value) 
                    for input_key, input_value in step_config["inputs"].items()
                }
            )
            # Needed for functions with kwargs
            additional_conf["inputs"] = {input_key:input_key for input_key in step_config["inputs"].keys()}
        else:
            injector = None
        
        step_node_config = {
            **config,
            **{c:v for c,v in step_config.items() if c not in ["inputs", "step_type"]},
            **additional_conf
        }
        with OverrideNodeExpander(fn, injector):
            step_nodes = base.resolve_nodes(fn, step_node_config)

        if len(step_nodes) != 1:
            raise ValueError("Steps can only contain pipeline elements that evaluate to a single node")

        (node_, ) = step_nodes
        return [node_.copy_with(
            name=step_name,
            tags={**node_.tags, **fm.recursive.NON_FINAL_TAGS}
        ),]


class SubModule(KwPrimitive):
    _NON_CONFIG_PARAMETERS = ["inputs", "module", "outputs", "external_inputs", "override_namespace_isolation"]

    @classmethod
    def _expand_python(
        cls,
        rule_name: str, 
        rule_config: typing.Dict[str, typing.Any]
    ):
        ## TODO
        # spec = importlib.util.spec_from_file_location("mod2.class2", "../mod2/class2.py")
        # foo = importlib.util.module_from_spec(spec)
        return creators.subdag_multi_out(
            importlib.import_module(rule_config["module"]),
            inputs = {k: infer_inject_parameter(v) for k,v in rule_config.get("inputs", {}).items()},
            config = {c:v for c,v in rule_config.items() if c not in cls._NON_CONFIG_PARAMETERS},
            namespace = rule_name,
            output_node_map = rule_config.get("outputs", {}),
            external_inputs = rule_config.get("external_inputs"),
            isolate_inputs = rule_config.get("override_namespace_isolation", False)
        )().generate_nodes()
        
    @classmethod
    def expand_nodes(
        cls,
        rule_name: str, 
        rule_config: typing.Dict[str, typing.Any], 
        function_map: typing.Dict[str, typing.Callable], 
        config: dict,
        **kwargs
    ):
        root_path = config.get("__global_root_base_path__", ".")
        module: str = rule_config["module"]
        if module.endswith(".hcl"):
            from lib.creators import config_dag
            import pygohcl
            @config_dag()
            def config_dag_fn():
                pass

            with open(os.path.join(root_path, module)) as fp:
                child_config = pygohcl.loads(fp.read())

            parent_outputs = rule_config.get("outputs")
            if parent_outputs is None: # No outputs defined on outer limits
                outputs = child_config.get("outputs")
                assert isinstance(outputs, dict), f"Rule: {rule_name}. An outputs map must be defined in module: {module} or in the calling module."
            else:
                child_outputs = child_config.get("outputs")
                if child_outputs is not None:
                    outputs = {}
                    for k,v in parent_outputs.items():
                        if v not in child_outputs:
                            raise ValueError(f"Rule: {rule_name}. Attempting to access variable {v} from module: {module}. However it is not exposed as an output.")
                        outputs[k] = child_outputs[v]
                else:
                    outputs = parent_outputs
            return base.resolve_nodes(config_dag_fn, {
                **child_config, 
                "inputs": {k: infer_inject_parameter(v) for k,v in rule_config.get("inputs", {}).items()},
                "outputs": outputs,
                # TODO maybe define input types from child config to allow type validation
                "namespace": rule_name, 
                "module": os.path.splitext(os.path.split(module)[0])[1],
                **{k: v for k, v in config.items() if k.startswith("__global_")}
            })

        # Load module from path (unfortunately a bit cumbersome)
        mod_loc = os.path.join(root_path, module)
        mod_name = "ham_internal_graph_modules."+mod_loc[:-3].replace("/", ".").replace("\\", ".")
        spec = importlib.util.spec_from_file_location(mod_name, mod_loc)
        dyn_mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = dyn_mod
        spec.loader.exec_module(dyn_mod)
        dyn_mod.__name__ = mod_name
        return creators.subdag_multi_out(
            dyn_mod,
            inputs = {k: infer_inject_parameter(v) for k,v in rule_config.get("inputs", {}).items()},
            config = {c:v for c,v in rule_config.items() if c not in cls._NON_CONFIG_PARAMETERS},
            namespace = rule_name,
            output_node_map = rule_config.get("outputs", {}),
            external_inputs = rule_config.get("external_inputs"),
            isolate_inputs = rule_config.get("override_namespace_isolation", False)
        ).generate_nodes(configuration={k: v for k, v in config.items() if k.startswith("__global_")})



class WhenTree(KwPrimitive):
    @classmethod
    def _create_additional_static_nodes(
        cls, nodes: typing.Collection[ham_node.Node], inputs, namespace: str
    ) -> typing.Collection[ham_node.Node]:
        # These already have the namespace on them
        # This allows us to inject values into the replayed subdag
        node_types = fm.recursive.extract_all_known_types(nodes)
        out = []
        for key, value in inputs.items():
            # TODO -- fix type derivation. Currently we don't use the specified type as we don't
            #  really know what it should be...
            new_node_name = fm.recursive.assign_namespace(key, namespace)
            if value.get_dependency_type() == fm.recursive.dependencies.ParametrizedDependencySource.LITERAL:
                out.append(
                    fm.recursive.create_static_node(
                        typ=fm.recursive.derive_type(value),
                        name=key,
                        value=value.value,
                        namespace=(namespace,),
                        tags=fm.recursive.NON_FINAL_TAGS,
                    )
                )
            elif value.get_dependency_type() == fm.recursive.dependencies.ParametrizedDependencySource.UPSTREAM:
                if new_node_name not in node_types:
                    continue
                out.append(
                    fm.recursive.create_identity_node(
                        from_=value.source,
                        typ=node_types[new_node_name],
                        name=key,
                        namespace=(namespace,),
                        tags=fm.recursive.NON_FINAL_TAGS,
                    )
                )
        return out


    @classmethod
    def expand_nodes(
        cls,
        tree_name: str, 
        tree_config: typing.Dict[str, typing.Any], 
        function_map: typing.Dict[str, typing.Callable], 
        config: dict,
        **kwargs
    ):
        when_tree, value_map, cond_map = builtin_funcs.WhenTree.from_config(tree_config)
        # TODO might need inputs here so that values listed internally can be identified to be bought into the namespace
        input_map = tree_config.get("inputs", {})
        # external_sources = list(input_map.keys())
        input_map_inferred = {k: infer_inject_parameter(v) for k,v in input_map.items()}

        # Handle all conditions
        nodes = []
        condition_external_inputs = {}
        for key, value in cond_map.items():
            if isinstance(value, builtin_funcs.ExprCondition):
                expr_node = base.resolve_nodes(
                    base_ops.expression, 
                    {"inputs": input_map,})[0]
                
                expr_node_injector = fm.inject(
                    **input_map_inferred,
                    expr=fm.value(value.expr),
                )

                nodes.append(
                    expr_node_injector.expand_node(
                        expr_node, {}, base_ops.expression
                    )[0].copy_with(name=key)
                )
            elif isinstance(value, builtin_funcs.ValueCondition):
                # Just create a new input mapping from external source
                condition_external_inputs[key] = fm.source(value.value)
                # external_sources.append(value.value)
        
        # Make a node to combine all inputs into a dictionary
        conditions_node = base.resolve_nodes(
            base_ops.collect_to_dict, 
            {"inputs": {k:k for k in cond_map.keys()},})[0]

        # Map external inputs into the conditions node      
        cond_node_inj = fm.inject(**condition_external_inputs)
        nodes.append(cond_node_inj.expand_node(conditions_node, {}, base_ops.collect_to_dict)[0].copy_with(
            name="__conditions__",
            typ=typing.Dict[str, typing.Union[bool, pd.Series]]
        ))

        # for k,v in value_map.items():
        #     if isinstance(v, fm.UpstreamDependency):
        #         external_sources.append(v.source)
        # input_map_inferred.update(value_map)
        # Handle all values:
        # for key, value in value_map.items():
        #     input_map[key] = infer_inject_parameter(value)


        # Make a node to combine all values into a dictionary
        values_node = base.resolve_nodes(
            base_ops.collect_to_dict, 
            {"inputs": {k:k for k in value_map.keys()},})[0]
        # Map external inputs into the values node      
        value_node_inj = fm.inject(**value_map)
        nodes.append(value_node_inj.expand_node(values_node, {}, base_ops.collect_to_dict)[0].copy_with(
            name="__variables__",
        ))

        output_node = ham_node.Node.from_fn(builtin_funcs.when_tree)
        output_node = output_node.copy_with(name=tree_name)
        
        output_node_injector = fm.inject(
            tree=fm.value(when_tree),
            variables=fm.source("__variables__"),
            conditions=fm.source("__conditions__"),
            upcast_config=fm.value(tree_config.get("upcast_config")),
        )
        # nodes.extend(base.resolve_nodes(when_tree_fn, {}))
        nodes.extend(output_node_injector.expand_node(output_node, config, builtin_funcs.when_tree))

        # Namespace nodes
        nodes = fm.subdag.add_namespace(
            nodes, 
            namespace=tree_name,
            inputs=input_map
        )

        nodes += cls._create_additional_static_nodes(nodes, input_map_inferred, tree_name)

        expose_output_externally = fm.recursive.create_identity_node(
            from_=fm.recursive.assign_namespace(tree_name,tree_name),
            typ=typing.Any,
            name=tree_name,
            namespace=tuple(),
            tags=fm.recursive.NON_FINAL_TAGS,
        )
        return nodes + [expose_output_externally]


        # def assign_ns_root_node(key):
        #     if key not in external_sources:
        #         return fm.recursive.assign_namespace(key, tree_name)
        #     return key


        # condition_input_types = {
        #     assign_ns_root_node(key): typing.Union[bool, pd.Series]
        #     for key in cond_map.keys()
        # }
        # # Not too much better can be done here
        # value_input_types = {
        #     assign_ns_root_node(key): typing.Any for key in value_map.keys()
        # }

        # def remapped_when_tree(
        #     _condition_keys=set(condition_input_types.keys()),
        #     _value_keys=set(value_input_types.keys()),
        #     _when_tree=when_tree,
        #     _upcast_config=tree_config.get("upcast_config"),
        #     **kwargs
        # ) -> typing.Any:
        #     return builtin_funcs.when_tree(
        #         tree=_when_tree,
        #         variables={k: kwargs[k] for k in _value_keys},
        #         conditions={k: kwargs[k] for k in _condition_keys},
        #         upcast_config=_upcast_config
        #     )
        # output_node = ham_node.Node(
        #     tree_name,
        #     typing.Any,
        #     builtin_funcs.when_tree.__doc__ if builtin_funcs.when_tree.__doc__ else "",
        #     callabl=remapped_when_tree,
        #     tags={"module": config.get("module", "undefined")},
        #     node_source=ham_node.NodeType.STANDARD,
        #     input_types={**condition_input_types, **value_input_types}
        # )
        # nodes.append(output_node)
        # return nodes

class ScoreCard(KwPrimitive):

    @classmethod
    def expand_nodes(
        cls,
        scorecard_name: str, 
        scorecard_config: typing.Dict[str, typing.Any], 
        function_map: typing.Dict[str, typing.Callable], 
        config: dict,
        **kwargs
    ):
        from logging import getLogger
        from lib.builtin_funcs.scorecard_impl import Score
        from lib.builtin_funcs.scorecard import run_scorecard
        scorecard_inner_conf = scorecard_config.get("data")
        if scorecard_inner_conf is None:
            assert "json" in scorecard_config, "Either json or data must be provided in scorecard"
            import json
            root_path = config.get("__global_root_base_path__", ".")
            with open(os.path.join(root_path, scorecard_config["json"])) as fp:
                scorecard_inner_conf = json.load(fp)
        score_model = Score(scorecard_inner_conf, getLogger(__name__))

        scorecard_node = Step.expand_nodes(
            step_name=scorecard_name,
            step_config= {
                "inputs": {
                    **scorecard_config.get("inputs", {}), 
                    "scorecard": fm.value(score_model), 
                    "include_pd": fm.value(scorecard_config.get("include_pd", True))
                },
                "step_type": "run_scorecard"
            },
            function_map={"run_scorecard": run_scorecard},
            config = {}
        )[0]
        # TODO finish node extract inputs
        outputs = scorecard_config.get("outputs", {})
        output_nodes = [scorecard_node]
        if len(outputs) > 0:
            # Note this stops one value being represented as multiple items
            inv_output_map = {v:k for k,v in outputs.items()}
            assert len(inv_output_map) == len(outputs), "Duplicate values detected in output map. This is not supported."
            column_extractor = fm.extract_columns(*[(k, pd.Series) for k in inv_output_map.keys()])
            output_nodes = column_extractor.transform_node(scorecard_node, {}, None)
            output_nodes = [
                n.copy_with(name=inv_output_map[n.name])
                if n.name in inv_output_map 
                else n 
                for n in output_nodes
            ]
        return output_nodes 

class ScoreTable(KwPrimitive):

    @classmethod
    def expand_nodes(
        cls,
        scorecard_name: str, 
        scorecard_config: typing.Dict[str, typing.Any], 
        function_map: typing.Dict[str, typing.Callable], 
        config: dict,
        **kwargs
    ):
        from lib.builtin_funcs.score_table import run_score_table
        scorecard_inner_conf = scorecard_config.get("data")
        if scorecard_inner_conf is None:
            assert "csv" in scorecard_config, "Either csv or data must be provided in scorecard"
            root_path = config.get("__global_root_base_path__", ".")
            scorecard_inner_conf = pd.read_csv(os.path.join(root_path, scorecard_config["json"]))
        else:
            scorecard_inner_conf = pd.DataFrame(scorecard_inner_conf)

        scorecard_node = Step.expand_nodes(
            step_name=scorecard_name,
            step_config= {
                "inputs": {
                    **scorecard_config.get("inputs", {}), 
                    "score_table_df": fm.value(scorecard_inner_conf),
                },
                "step_type": "run_scorecard"
            },
            function_map={"run_scorecard": run_score_table},
            config = {}
        )[0]
        outputs = scorecard_config.get("outputs", {})
        output_nodes = [scorecard_node]
        if len(outputs) > 0:
            # Note this stops one value being represented as multiple items
            inv_output_map = {v:k for k,v in outputs.items()}
            assert len(inv_output_map) == len(outputs), "Duplicate values detected in output map. This is not supported."
            column_extractor = fm.extract_columns(*[(k, pd.Series) for k in inv_output_map.keys()])
            output_nodes = column_extractor.transform_node(scorecard_node, {}, None)
            output_nodes = [
                n.copy_with(name=inv_output_map[n.name])
                if n.name in inv_output_map 
                else n 
                for n in output_nodes
            ]
        return output_nodes 



PRIMITIVES: typing.Dict[str, typing.Union[Primitive, KwPrimitive]] = {
    "step": Step,
    "external": SubModule,
    "tree": WhenTree,
    "scorecard": ScoreCard,
    "scoretable": ScoreTable
}