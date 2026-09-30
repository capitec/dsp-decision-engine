# `each`: per-row vs batch, before unifying

Baseline for the decision to collapse `EachMode.PER_ROW` and `EachMode.BATCH`
into one scatter/gather primitive. Reproduce with:

    uv run python benchmarks/each_modes.py

Full matrix (items 1/5/20/100 × records 1–10000 × light/complex × per_row/batch)
is in `benchmarks/results_each_modes.csv`. `fused` engine mode throughout.

Two children: **light** is one scalar step (`weight > heavy_kg`); **complex** is
five chained steps then a 4-iteration `loop`.

## The shape

- **per_row** is flat: ~170–270 µs/record (light), ~266 µs/record (complex),
  scaling linearly with items-per-record. No batch benefit, no fixed cost.
- **batch** has a ~3–4 ms fixed cost (the nested `Engine().bind` plus the
  polars explode/group_by/join round-trip) but ~20× better marginal cost:
  ~7.5–8.8 µs/record at 10k records.
- **Crossover** is ~10–50 records, roughly independent of child complexity and
  items-per-record.

## items = 5 (µs to run the frame)

| child | records | per_row (µs) | batch (µs) | batch/per_row |
|---|---|---|---|---|
| light | 1 | 382 | 4,199 | 11.00x |
| light | 10 | 2,005 | 3,812 | 1.90x |
| light | 50 | 9,107 | 4,870 | 0.53x |
| light | 100 | 18,071 | 4,721 | 0.26x |
| light | 500 | 86,987 | 7,887 | 0.09x |
| light | 1,000 | 174,857 | 12,082 | 0.07x |
| light | 10,000 | 1,730,346 | 75,328 | 0.04x |
| light | µs/rec @ 10k | 173.0 | 7.5 | |
| complex | 1 | 419 | 3,944 | 9.41x |
| complex | 10 | 2,863 | 4,821 | 1.68x |
| complex | 50 | 13,349 | 4,712 | 0.35x |
| complex | 100 | 26,666 | 5,455 | 0.20x |
| complex | 500 | 132,562 | 9,189 | 0.07x |
| complex | 1,000 | 264,570 | 14,245 | 0.05x |
| complex | 10,000 | 2,660,185 | 87,556 | 0.03x |
| complex | µs/rec @ 10k | 266.0 | 8.8 | |

## items = 20 (µs to run the frame)

| child | records | per_row (µs) | batch (µs) | batch/per_row |
|---|---|---|---|---|
| light | 1 | 749 | 3,427 | 4.57x |
| light | 10 | 5,625 | 4,121 | 0.73x |
| light | 50 | 27,045 | 5,828 | 0.22x |
| light | 100 | 53,379 | 6,504 | 0.12x |
| light | 500 | 264,597 | 14,814 | 0.06x |
| light | 1,000 | 516,684 | 24,825 | 0.05x |
| light | 10,000 | 5,211,430 | 210,175 | 0.04x |
| light | µs/rec @ 10k | 521.1 | 21.0 | |
| complex | 1 | 1,179 | 4,883 | 4.14x |
| complex | 10 | 9,789 | 5,312 | 0.54x |
| complex | 50 | 47,937 | 5,956 | 0.12x |
| complex | 100 | 95,534 | 6,858 | 0.07x |
| complex | 500 | 475,966 | 16,314 | 0.03x |
| complex | 1,000 | 953,984 | 28,369 | 0.03x |
| complex | 10,000 | 9,572,287 | 217,094 | 0.02x |
| complex | µs/rec @ 10k | 957.2 | 21.7 | |

## What this means for unification

Unifying on "always explode and run over element rows" is only safe if the
primitive drops batch's ~4 ms fixed cost. That cost is the nested
`Engine().bind` and the polars explode/group_by/join round-trip, not the map
itself — an in-kernel ragged scatter/gather should land single-record near
per_row (~400–500 µs) while keeping batch's ~8 µs/record tail. If it does,
one mode strictly dominates; if it still pays a few ms to set up, keep both.

The per_row tail is also worth noting: it is per-item Python/numba dispatch, so
it scales with items, not records — that is the number a compiled
scatter/gather must beat at every items count.

## The unified primitive (`ScatterGatherNode`)

Implemented `each` as a first-class `ScatterGatherNode` (the child is a real IR
child; the runner scatters the list into element rows, runs the child over them,
gathers back). Data: `benchmarks/results_each_unified.csv`.

### items = 5 (µs to run the frame)

| child | records | per_row | batch | unified | unified vs per_row |
|---|---|---|---|---|---|
| light | 1 | 382 | 4,199 | 219 | 1.74x |
| light | 10 | 2,005 | 3,812 | 428 | 4.68x |
| light | 50 | 9,107 | 4,870 | 1,319 | 6.90x |
| light | 100 | 18,071 | 4,721 | 2,496 | 7.24x |
| light | 500 | 86,987 | 7,887 | 11,912 | 7.30x |
| light | 1,000 | 174,857 | 12,082 | 31,168 | 5.61x |
| light | 10,000 | 1,730,346 | 75,328 | 249,133 | 6.95x |
| complex | 1 | 419 | 3,944 | 473 | 0.89x |
| complex | 10 | 2,863 | 4,821 | 1,261 | 2.27x |
| complex | 50 | 13,349 | 4,712 | 4,827 | 2.77x |
| complex | 100 | 26,666 | 5,455 | 9,323 | 2.86x |
| complex | 500 | 132,562 | 9,189 | 43,520 | 3.05x |
| complex | 1,000 | 264,570 | 14,245 | 89,290 | 2.96x |
| complex | 10,000 | 2,660,185 | 87,556 | 904,101 | 2.94x |

### items = 20 (µs to run the frame)

| child | records | per_row | batch | unified | unified vs per_row |
|---|---|---|---|---|---|
| light | 1 | 749 | 3,427 | 257 | 2.92x |
| light | 10 | 5,625 | 4,121 | 864 | 6.51x |
| light | 50 | 27,045 | 5,828 | 3,676 | 7.36x |
| light | 100 | 53,379 | 6,504 | 7,083 | 7.54x |
| light | 500 | 264,597 | 14,814 | 34,668 | 7.63x |
| light | 1,000 | 516,684 | 24,825 | 69,326 | 7.45x |
| light | 10,000 | 5,211,430 | 210,175 | 754,505 | 6.91x |
| complex | 1 | 1,179 | 4,883 | 680 | 1.73x |
| complex | 10 | 9,789 | 5,312 | 3,401 | 2.88x |
| complex | 50 | 47,937 | 5,956 | 15,247 | 3.14x |
| complex | 100 | 95,534 | 6,858 | 30,018 | 3.18x |
| complex | 500 | 475,966 | 16,314 | 150,422 | 3.16x |
| complex | 1,000 | 953,984 | 28,369 | 290,512 | 3.28x |
| complex | 10,000 | 9,572,287 | 217,094 | 3,048,526 | 3.14x |

## Verdict

The primitive does what unification wanted: it drops batch's ~4 ms fixed cost
(single-record is ~219–473 µs, roughly per_row and ~9x better than batch) while
the child body stays compiled. It is a strict win over per_row at every point
from ~10 records up (2–7x), and over batch up to ~50 records.

The remaining gap is batch's *tail*: on wide frames unified is still 3–10x
slower than batch. The cause is not the map (the child steps compile; `fallbacks()` is
empty) but the scatter/gather **driver** — the Python loop that explodes the
list into flat per-field arrays and rebuilds the enriched list per element
(`decider/engine/run/runners/interpreted.py::_scatter_gather`). To close it, make
that explode/gather vectorised (a compiled gather into a list column, or polars
explode/unnest like batch used) rather than per-element Python. Until then,
one mode is not strictly dominant on wide frames; single-record — the documented
priority — already favours the primitive.

