"""One int64 envelope over every evidence type: table rows, tree paths, scorecard bands,
reasons, overrides, and numeric value evidence.

The envelope is one int64 per event. Fixed-width numeric evidence rides in it
directly (as an index, the existing precedent for trees). Strings never enter
the kernel: a `reason` or `rule name` is an index into a per-plan constant table,
decoded in Python. A float64 value does not *fit* the int64's bit fields, but its
64 bits do: reinterpretation round-trips a float64 losslessly. Overrides are a
kind tag plus a versioned companion payload (the value).

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/evidence_types.py
"""
import numpy as np

STEP_BITS, KIND_BITS, ARM_BITS, ITER_BITS = 20, 8, 16, 24
STEP_SHIFT = KIND_BITS + ARM_BITS + ITER_BITS
KIND_SHIFT = ARM_BITS + ITER_BITS
ARM_SHIFT = ITER_BITS

# Kinds (8-bit tag). The companion payload/constant table is versioned per plan.
STEP, TABLE_ROW, TREE_PATH, SCORECARD_BAND, REASON, OVERRIDE, VALUE = range(7)

# A per-plan constant table, versioned: reasons and rule names never enter the kernel.
CONST_TABLE = ["income_too_low", "credit_history_thin", "over_limit"]


def pack(step_ref, kind, arm, iteration):
    return (step_ref << STEP_SHIFT) | (kind << KIND_SHIFT) | (arm << ARM_SHIFT) | iteration


def payload_for(kind, value):
    """The companion payload for evidence whose value doesn't fit the envelope's index bits."""
    if kind == VALUE:  # a float64: reinterpret bits, round-trips losslessly
        return int(np.float64(value).view(np.int64))
    if kind == OVERRIDE:  # a value, versioned in a companion buffer
        return value
    return value  # indices into the constant table


if __name__ == "__main__":
    e = pack(0, TABLE_ROW, 0, 0)
    print(f"table row: matched row index 2 fits as the event's arm/iteration (or a dedicated index); "
          f"the row's label is CONST_TABLE[k], decoded in Python")

    e = pack(0, TREE_PATH, 0, 0)
    print(f"tree path: the walker already emits one int64 'path number' ({e & 0xFF} kind, path in arm/iter); "
          f"the '>'-joined node ids stay a constant table (tree trace.py precedent)")

    e = pack(0, SCORECARD_BAND, 0, 0)
    print(f"scorecard band: band index 0..n-1 fits as a small int; band label is a constant table")

    e = pack(0, REASON, 2, 0)  # arm field = index into CONST_TABLE
    print(f"reason: index 2 -> {CONST_TABLE[2]!r}; the string never enters the kernel, only the index")

    val = 42.125
    bits = payload_for(VALUE, val)
    back = np.array([np.int64(bits)], dtype=np.int64).view(np.float64)[0].item()
    assert back == val, "float64 bits must round-trip"
    print(f"numeric value evidence: {val} -> int64 {bits} (bit reinterpretation) -> {back}; "
          f"a float64's 64 bits fit one int64 losslessly, so a value can ride the envelope")

    e = pack(0, OVERRIDE, 0, 0)
    print(f"override: kind tag OVERRIDE + a versioned companion payload holding the value; "
          f"the Session.set producer 'override@<path>' already names it")

    print("conclusion: table rows, tree paths, scorecard bands, reasons and overrides are all small-int "
          "indices into a versioned constant table; a float64 value is bit-reinterpreted, not field-packed")
