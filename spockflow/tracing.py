from typing import Any

# TODO expand for dataframes

class TracerSeriesFunction:
    def __init__(self, name, caller_series):
        self.name = name
        self.caller_series = caller_series
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return TracerSeries(f"{self.name}({self.caller_series})")

class TracerSeries:
    def __init__(self, name, is_base=False):
        self._name = name
        self.is_base = is_base
    @staticmethod
    def _get_other_name(value) -> str:
        if isinstance(value, TracerSeriesFunction): return "Unknown Type"
        if isinstance(value, TracerSeries): return value.name
        if isinstance(value, (float, int)): return str(value)
        if isinstance(value, str): return value
        if isinstance(value, (list, tuple, set)):
            if (len(value) <= 5) and (all((isinstance(v, (str, float, int)) for v in value))):
                return f"<{','.join(value)}>"
            else:
                return "List"
        return repr(value)
    
    @property
    def name(self):
        if self.is_base: return self._name
        return f"({self._name})"
    
    def __ceil__(self) -> "TracerSeries":
        return TracerSeries(f"⌊{self.name}⌋")
    def __floor__(self) -> "TracerSeries":
        return TracerSeries(f"⌈{self.name}⌉")
    
    def __invert__(self) -> "TracerSeries":
        return TracerSeries(f"!{self.name}")
    def __abs__(self) -> "TracerSeries":
        return TracerSeries(f"|{self.name}|")
    def __neg__(self) -> "TracerSeries":
        return TracerSeries(f"-{self.name}")
    def __pos__(self) -> "TracerSeries":
        return TracerSeries(f"+{self.name}")
    
    def __and__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name}&{self._get_other_name(__value)}")
    def __rand__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)}|{self.name}")
    def __or__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name}&{self._get_other_name(__value)}")
    def __ror__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)}|{self.name}")
    
    def __xor__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} xor {self._get_other_name(__value)}")
    def __rxor__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} xor {self.name}")
    
    def __sub__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} - {self._get_other_name(__value)}")
    def __rsub__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} + {self.name}")
    
    def __add__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} - {self._get_other_name(__value)}")
    def __radd__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} + {self.name}")
    
    def __mul__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} x {self._get_other_name(__value)}")
    def __rmul__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} x {self.name}")
    
    def __matmul__(self, other) -> "TracerSeries":
        return TracerSeries(f"{self.name} . {self._get_other_name(other)}")
    def __rmatmul__(self, other) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(other)} . {self.name}")
    
    def __floordiv__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} // {self._get_other_name(__value)}")
    def __rfloordiv__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} // {self.name}")
    
    def __div__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} / {self._get_other_name(__value)}")
    def __rdiv__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} / {self.name}")
    
    def __truediv__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} / {self._get_other_name(__value)}")
    def __rtruediv__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} / {self.name}")
    
    def __mod__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} % {self._get_other_name(__value)}")
    def __rmod__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} % {self.name}")
    
    # def __divmod__(self, __value) -> "TracerSeries":
    #     return TracerSeries(f"{self.name} % {self._get_other_name(__value)}")
    # def __rdivmod__(self, __value) -> "TracerSeries":
    #     return TracerSeries(f"{self._get_other_name(__value)} % {self.name}")


    def __le__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} ≤ {self._get_other_name(__value)}")
    def __rle__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} ≤ {self.name}")
    def __lt__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} < {self._get_other_name(__value)}")
    def __rlt__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} < {self.name}")
    def __ge__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} ≥ {self._get_other_name(__value)}")
    def __rge__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} ≥ {self.name}")
    def __gt__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} > {self._get_other_name(__value)}")
    def __rgt__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} > {self.name}")
    def __eq__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self.name} = {self._get_other_name(__value)}")
    def __req__(self, __value) -> "TracerSeries":
        return TracerSeries(f"{self._get_other_name(__value)} = {self.name}")
    def __getattr__(self, name):
        return TracerSeriesFunction(name, self.name)

