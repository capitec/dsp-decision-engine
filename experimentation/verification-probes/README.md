# Verification probes

Two claims from the overnight string/type strands were load-bearing enough to
check directly rather than accept from the strand that produced them. These are
those checks, standalone and reproducible.

```
../../.venv/bin/python probe_decider2_refcount.py
../../.venv/bin/python probe_category_mask.py
```

## `probe_decider2_refcount.py`

`experimentation/numba-c-string-matching/` found that writing a tree's row body
as a *separate* numba function taking arrays costs ~365 ns/row, because numba
emits NRT reference-counting traffic for every array argument on every row. That
is roughly the shape of decider2's own `trees/interpreter.py:walk_tree`, so the
question is whether decider2 pays it today.

It does not:

```
A  decider2 shape (per-row call into walk_tree, 7 array args) : 45.64 ns/row  NRT (4, 11)
B  same body inlined in one loop, arrays captured             : 43.80 ns/row  NRT (0, 7)
ratio A/B = 1.04x
```

decider2's structure arrays are loop-invariant, so the bookkeeping hoists out of
the loop. The trap stays live for a **typed** walker, where per-row feature
arrays would be passed in and are not invariant — keep per-row inputs as scalars
or raw integers.

## `probe_category_mask.py`

Both string strands recommend the per-category mask (run the pattern over the
distinct values once, then one array lookup per row). This measures it end to
end on 1M rows over 12 distinct values, including the costs the headline figure
leaves out:

```
per-row mask lookup in kernel            :  0.94 ns/row
total, column already dictionary-encoded :  1.07 ns/row
polars str.contains over every row       : 34.09 ns/row
total, if you must encode it yourself    : 50.53 ns/row
```

The 32x win holds **only** while the column arrives already encoded — which
decider2 does at the boundary today, and which is the precondition to re-check
before adopting the mask.
