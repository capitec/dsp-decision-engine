import typing as t

from ped.modules.core import PEDNode
from .core import BaseAdapter

class TypeAdapter(BaseAdapter):
    type: t.Literal['type_adapter']
    def adapt(
        self, 
        inputs: t.List[PEDNode],
    ) -> t.List[PEDNode]:
        return inputs
        # nodes_dict = {node.name: node for node in inputs}

        # node_output_types = ... Get the node output types

        # for node in inputs:
        #     for input in node inputs
        #         if input in node_output_types
        #             if types dont match add a new node maybe something like {node_name}.__converted_to__{expected_type} and add it to the graph and change the input of the original node to be this new node instead of the original node that was not matching.
        #             # You will need to check that it doesnt already exist in the node output types

# im hoping for a few things with the conversion
# 1. if the input type is a pl.Expr like def test(x: pl.Expr) and its not from somewhere in the graph then i think we assume that its coming from a lazy frame called input
# So in the above example i want to make a new node called input.__extract_x__ or something like that that does something like input.select('x') or we just make a column called 'x' and pass it into the funciton

# but maybe the input is something like "input2.x: pl.Expr" and if input2 or input2.x isnt in the graph then im hoping we can make a node x = input2.select('x2')
# It might be stupid but i want the user to by default run the flow with flow.Execute(input=frame1, input2=frame2) and it should return a lazy frame or frames.