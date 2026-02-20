import typing as t
from ped.types import TInputType
from ..graph import BaseGraph
from dataclasses import dataclass
if t.TYPE_CHECKING:
    from hamilton.driver import Driver

@dataclass
class HamiltonGraph(BaseGraph):
    hamilton_driver: "Driver"
    default_outputs: t.List[str]
    
    def execute(
        self, 
        inputs: TInputType,
        output_overrides: t.Optional[t.List[str]] = None,
        overrides: t.Optional[t.Dict[str, t.Any]] = None,
        display_graph: bool = False,
    ) -> t.Any:
        """Execute the graph and return the output."""
        return self.hamilton_driver.execute(
            inputs=inputs, 
            final_vars=output_overrides or self.default_outputs, 
            overrides=overrides, 
            display_graph=display_graph,
        )
