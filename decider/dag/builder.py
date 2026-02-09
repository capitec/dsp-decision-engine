import typing as t
from hamilton.driver import Builder

from .expanders.base import DeciderExpandableModule
from .core import DeciderAdaptorHook
from decider.typing import inherit_signature_from


class DeciderBuilder(Builder):
    @inherit_signature_from(Builder.__init__)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._decider_adaptor = DeciderAdaptorHook()
        self.adapters = [self._decider_adaptor]

    def include(
        self, 
        module: DeciderExpandableModule, 
        namespace: str|None = None, 
        parameter_mapping: t.Dict[str, str] = None
    ) -> "t.Self":
        if parameter_mapping is not None:
            from .expanders.inject import InjectedModule
            module = InjectedModule(
                parameter_mapping=parameter_mapping, 
                expander=module
            )
        if namespace is not None:
            from .expanders.subdag import NamespacedModule
            module = NamespacedModule(
                namespace=namespace, 
                expander=module
            )
        self._decider_adaptor.add_module(module)
        return self
    
    def compile(self):
        from .compile import CompiledModulePlaceholder
        return CompiledModulePlaceholder(None)
