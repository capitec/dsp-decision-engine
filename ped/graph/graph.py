from abc import ABC, abstractmethod
from ped.types import TInputType, TOutputType

class BaseGraph(ABC):
    @abstractmethod
    def execute(
        self, 
        inputs: TInputType,
        **kwargs,
    ) -> TOutputType:
        """Execute the graph and return the output."""
        ...