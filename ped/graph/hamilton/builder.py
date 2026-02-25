import typing as t
from hamilton.driver import Builder
from pydantic import PrivateAttr
from .adaptor import DeciderAdaptorHook
from ..builder import BaseBuilder
from .graph import HamiltonGraph

if t.TYPE_CHECKING:
    from ped.modules import ConstructedGraphModules


class HamiltonBuilder(BaseBuilder[HamiltonGraph]):
    type: t.Literal["hamilton"] = "hamilton"

    def build_graph(
        self, 
        modules: "ConstructedGraphModules", 
        output_nodes: t.List[str],
    ) -> HamiltonGraph:
        adaptor = DeciderAdaptorHook(modules=modules)
        hamilton_builder = (
            Builder()
            # TODO we must maybe add some default result builders here
            .with_adapter(adaptor)
        )
        hamilton_graph = hamilton_builder.build()
        return HamiltonGraph(hamilton_graph, default_outputs=output_nodes)
