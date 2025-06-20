## Spockflow Architecture Documentation - Landing Page


### **Directory Structure**

To better understand Spockflow’s architecture, let’s explore the key folders and their responsibilities within the package:

---

#### **1. `spockflow/`**
This is the core of the Spockflow framework, containing all of its primary modules and components.

---

##### **core.py**
- The main entry point for Hamilton integration.
- Defines a custom decorator for injecting Spockflow logic into Hamilton’s DAG framework.
- Expands Hamilton subdags with configurable components and generates nodes from these components.
- Calls `initialize_spock_module` to inject the Spockflow functionality into a given module, allowing for the automatic generation of Hamilton nodes.

Example of Hamilton subdag integration:
```python
@subdag(
    feature_modules,
    inputs={"path": source("source_path")},
    config={}
)
def feature_engineering(feature_df: pd.DataFrame) -> pd.DataFrame:
    return feature_df
```
- In Spockflow, the `initialize_spock_module` decorator ensures that subdags are expanded and executed according to the framework's configuration.

---

##### **nodes.py**
- Contains the definition of `VariableNode`, the core class responsible for transforming configuration-driven logic into executable Hamilton nodes.
- Handles utilities such as `CloneVariableNode` (for duplicating nodes) and `AliasedVariableNode` (for renaming nodes without re-executing).
- Uses Pydantic classes to serialize and deserialize configuration, making it easier to manage node definitions and configurations.
- The `generate_nodes` function within `VariableNode` handles the actual creation of subnodes, ensuring that each node can be expanded within a Hamilton DAG.
  
```python
def _generate_nodes(self, ...):
    ...
    node_functions = inspect.getmembers(
        compiled_variable_node, predicate=self._does_define_node
    )
    ...
```
This method identifies and expands functions within a module as Hamilton nodes, ensuring that subcomponents can be injected into larger data pipelines.

---

##### **_serializable.py**
- Provides utilities to help with the serialization and deserialization of data, particularly for handling Pandas DataFrames and Series.
- Ensures that data passed through Spockflow nodes can be properly transformed and maintained across different steps of the pipeline.

---

#### **2. `components/`**
Contains all the core components and decision-oriented modules in Spockflow, including:
- **Decision Trees**: Build decision trees to enforce rules for data enrichment and transformations.
- **Scorecards**: Create scoring systems for evaluating data based on multiple parameters.
- **Decision Tables**: Define mappings of input values to outputs based on set conditions.

Each of these components is built as reusable modules that can be configured and inserted into your data flows.

---

#### **3. `inference/`**
- Contains logic and tools to serve models via endpoints compatible with services like AWS SageMaker.

---


### **How to Define a Custom Node in Spockflow**

In Spockflow, custom nodes allow users to extend the framework's functionality by creating new components that integrate seamlessly into the Hamilton DAG-based architecture. A custom node is a class that inherits from `VariableNode` and defines its own behavior for node creation, input handling, and execution.

Here, we'll define a custom `Tree` node as an example of how to create a custom decision-making process using Spockflow’s infrastructure.

#### **Step 1: Define the Custom Node Class**

To create a custom node, you need to subclass `VariableNode` and define several key components, such as input fields, the `compile()` method, and custom logic for handling inputs and outputs.

```python
class Tree(VariableNode):
    # This is used in visualisation by Hamilton
    doc: str = "This executes a user-defined decision tree"
    
    # Define fields using Pydantic (these can be any fields for configuration)
    execution_conditions: typing.List[str]
    execution_outputs: typing.List[str]
    
    # The compile function needs to be provided. By default, it will just return self.
    def compile(self):
        # This step may involve transforming or processing the input data into a usable format
        from .compiled import CompiledNumpyTree
        return CompiledNumpyTree(self)
```

- `execution_conditions` and `execution_outputs` are lists of strings that define the conditions and outputs associated with the decision tree.
- The `compile()` function is responsible for transforming the raw input data into a format that can be used by the Hamilton DAG. In this case, it initializes a `CompiledNumpyTree`.

#### **Step 2: Define a Compiled Representation for the Node**

To optimize how the node’s logic is executed, we can define a compiled version of the node, such as `CompiledNumpyTree`. This compiled version will contain the logic to handle the execution and manage inputs dynamically.

```python
class CompiledNumpyTree:
    def __init__(self, tree: Tree) -> None:
        # This constructor will process and configure the tree into an executable form
        self.tree = tree
        # Additional processing logic for the tree can go here
    
    def _get_inputs(self, function: typing.Callable):
        # Returns the expected input types for the node
        node_input_types = {o: pd.DataFrame for o in self.tree.execution_outputs}
        node_input_types.update({c: typing.Union[np.ndarray, pd.Series] for c in self.tree.execution_conditions})
        return node_input_types
```

- The `CompiledNumpyTree` class is responsible for transforming the raw `Tree` object into an optimized version that can be used in a Hamilton DAG.
- The `_get_inputs` function dynamically determines the input types required for this node’s execution.

#### **Step 3: Define Node Functions with `@creates_node`**

Next, we define the various operations that make up the logic of our custom `Tree` node. These operations are implemented as functions within the `Tree` class and are decorated with `@creates_node`. The `@creates_node` decorator tells Spockflow to treat these methods as subnodes within the Hamilton DAG.

```python
from spockflow.nodes import creates_node
import numpy as np
import pandas as pd

class Tree(VariableNode):
    # Other fields and compile method defined previously

    @creates_node(kwarg_input_generator="_get_inputs")  # Generates node inputs dynamically
    def format_inputs(
        self, **kwargs: typing.Union[pd.DataFrame, pd.Series]
    ) -> TFormatData:
        # Process inputs and return transformed data
        pass

    @creates_node()  # Defines a subnode for conditions met
    def conditions_met(self, format_inputs: TFormatData) -> np.ndarray:
        # Logic for evaluating conditions based on inputs
        pass

    @creates_node()  # Defines a subnode for prioritizing conditions
    def prioritized_conditions(self, conditions_met: np.ndarray) -> np.ndarray:
        # Logic for prioritizing conditions
        pass

    @creates_node()  # Defines a subnode for generating condition names
    def condition_names(self, format_inputs: TFormatData) -> typing.List[str]:
        # Logic to generate the names of the conditions
        pass

    @creates_node()  # Defines a subnode for the final decision logic
    def all(
        self,
        format_inputs: TFormatData,
        conditions_met: np.ndarray,
    ) -> pd.DataFrame:
        # Logic for making a decision based on inputs and conditions
        pass

    @creates_node(is_namespaced=False)  # This node will be created outside the namespace
    def get_results(
        self,
        format_inputs: TFormatData,
        prioritized_conditions: np.ndarray,
    ) -> pd.DataFrame:
        # Final output of the decision tree process
        pass
```

- **`@creates_node`**: This decorator defines the function as a subnode in the DAG.
  - The `kwarg_input_generator="_get_inputs"` argument is used to specify how to dynamically determine the input types for this node.
  - Each method, such as `format_inputs()`, `conditions_met()`, etc., corresponds to a specific operation in the decision tree.
  
The above tree when created as follows:

```python
# Example Tree node instance
example_tree = Tree(execution_conditions=["a", "b"], execution_outputs=["c", "d"])

```
will create the following DAG:
![](./tree.drawio.svg)

- The above relationships represent the connections between nodes, where each `@creates_node` function becomes part of the Hamilton DAG.
- The `example_tree.format_inputs` node takes inputs `a`, `b`, `c`, and `d`, and feeds them into subsequent nodes like `conditions_met`, `prioritized_conditions`, and others.

---

### **Summary of Custom Node Creation Steps**

1. **Define the Node Class**: Inherit from `VariableNode` and specify fields such as conditions and outputs.
2. **Compile the Node**: Provide a `compile()` method to transform the node into an optimized executable form (e.g., `CompiledNumpyTree`).
3. **Define Operations as Subnodes**: Use `@creates_node` to define methods that represent different parts of the decision-making process.
4. **Establish Dependencies**: The created subnodes will automatically link based on their input/output relationships, forming a complete DAG.

By following these steps, you can define complex, decision-oriented nodes in Spockflow and integrate them seamlessly into a Hamilton-based data pipeline.