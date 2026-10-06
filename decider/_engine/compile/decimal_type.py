# A `Decimal` in a kernel is an int64 raw count, typed by its scale alone: precision lives in
# annotations, and the units string never reaches a kernel or a cache key. Columns, params
# bundles and outputs stay plain int64 and are typed at the kernel's edge, because a Python
# Decimal crossing the dispatcher costs microseconds (a tuple of them tens). The operators
# live in `semantics/decimals.py`. Named `decimal_type`, not `decimal`: a directory on
# `sys.path` would shadow the stdlib module.
from __future__ import annotations

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel import models
from numba.core.typeconv import Conversion
from numba.extending import NativeValue, box, lower_cast, register_model, typeof_impl, unbox

from decider.ops.decimal import Decimal


class DecimalType(types.Type):
    def __init__(self, scale: int) -> None:
        self.scale = scale
        super().__init__(f"decimal[{scale}]")

    @property
    def key(self) -> int:
        return self.scale

    def unify(self, typingctx, other):
        # `total = 0; total += d` and `x if c else y` over two scales take the larger scale.
        if isinstance(other, DecimalType):
            return self if self.scale >= other.scale else other
        if isinstance(other, types.Integer):
            return self

    def can_convert_to(self, typingctx, other):
        if isinstance(other, DecimalType) and other.scale >= self.scale:
            return Conversion.safe

    def can_convert_from(self, typingctx, other):
        if isinstance(other, types.Integer):
            return Conversion.safe


@register_model(DecimalType)
class _DecimalModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(64))


@typeof_impl.register(Decimal)
def _typeof_decimal(val, c):
    return DecimalType(val.scale)


@unbox(DecimalType)
def _unbox(typ, obj, c):
    raw = c.pyapi.object_getattr_string(obj, "raw")
    value = c.pyapi.long_as_longlong(raw)
    c.pyapi.decref(raw)
    return NativeValue(value, is_error=cgutils.is_not_null(c.builder, c.pyapi.err_occurred()))


@box(DecimalType)
def _box(typ, val, c):
    cls = c.pyapi.unserialize(c.pyapi.serialize_object(Decimal))
    make = c.pyapi.object_getattr_string(cls, "from_raw")
    out = c.pyapi.call_function_objargs(make, (c.pyapi.long_from_longlong(val), c.pyapi.long_from_longlong(ir.Constant(ir.IntType(64), typ.scale))))
    c.pyapi.decref(make)
    c.pyapi.decref(cls)
    return out


@lower_cast(types.Integer, DecimalType)
def _from_int(context, builder, fromty, toty, val):
    return builder.mul(context.cast(builder, val, fromty, types.int64), ir.Constant(ir.IntType(64), 10 ** toty.scale))


@lower_cast(DecimalType, DecimalType)
def _rescale_up(context, builder, fromty, toty, val):
    return builder.mul(val, ir.Constant(ir.IntType(64), 10 ** (toty.scale - fromty.scale)))
