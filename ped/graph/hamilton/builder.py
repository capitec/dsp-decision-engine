import typing as t
from hamilton.driver import Builder
from pydantic import PrivateAttr
from .adapter import DeciderAdapterHook
from ..builder import BaseBuilder
from .graph import HamiltonGraph

if t.TYPE_CHECKING:
    from ped.modules import ConstructedGraphModules


class HamiltonBuilder(BaseBuilder[HamiltonGraph]):
    type: t.Literal["hamilton"]

    def build_graph(
        self, 
        modules: "ConstructedGraphModules", 
        output_nodes: t.List[str],
    ) -> HamiltonGraph:
        adapter = DeciderAdapterHook(modules=modules)
        hamilton_builder = (
            Builder()
            # TODO we must maybe add some default result builders here
            .with_adapter(adapter)
        )
        hamilton_graph = hamilton_builder.build()
        return HamiltonGraph(hamilton_graph, default_outputs=output_nodes)
