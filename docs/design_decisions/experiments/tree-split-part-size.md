# How wide should the parts of a split tree be?

Date: 2026-10-09 · Script: `benchmarks/tree_split_sweep.py` · Raw cells: `tree-split-part-size.jsonl` (second session last)

## Purpose
A `mode: "all"` tree of more than 32 outputs ran as parts of 32 (see [wide output compile](wide-output-compile.md)). A single record then calls several walkers, which costs 11-14% of `score()` at 50-250 rules. Which part size keeps `score()` close to an unsplit tree without bringing the compile time back?

## Method
Ten rule shapes, one boolean output per rule: single-condition thresholds, `and` of three (12 features; also 4 and 40 features), `or` composites, `between`, `isin` over numbers, exact string matches (a pattern list and a single pattern), and a mix with the shape counts of `examples/transaction-fraud`'s rule set (and3 40%, and2 35%, or2 12%, and5 5%, 20 features, a few computed ratios). Each at 50, 100, 250 and 521 rules and part sizes 16, 32, 64, 128, 256 and unsplit (a size of at least the rule count is the unsplit tree; unsplit was skipped at 521 except for `and3`, 4 minutes cold).

Per cell, two fresh processes sharing one `NUMBA_CACHE_DIR`: the first is cold (bind plus first `score()`), the second warm (disk cache). Each then times 3000 `score()` calls after 1000 warm-up calls (p50, p99) and the best of three `run()` over 200k rows. Cells were run in sequence in one session on a loaded box; `light.sh` serialised them against other agents. A second session re-ran `and3`, `fraud`, `isin_num` and `threshold` at 100 and 250 rules in the reverse order of sizes.

Noise: the p50 of the cold and warm process of one cell differs by a median 1% (p90 5%, max 18%); the same cell in two sessions by a median 6% (max 18%). Differences under about 6% are not claimed.

## Result
`score()` p50 against the unsplit tree (at 521 rules against 256); median over the ten shapes, range in brackets:

| Rules | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 50 | 1.25 | 1.11 | whole | whole | whole |
| 100 | 1.34 | 1.18 | 1.08 (1.03-1.21) | whole | whole |
| 250 | 1.28 | 1.13 | 1.03 (0.96-1.16) | 1.03 (0.97-1.11) | whole |
| 521 | 1.27 | 1.11 | 1.02 (0.81-1.08) | 0.98 (0.85-1.07) | 1.00 |

Cold bind plus first score, median seconds (range):

| Rules | 32 | 64 | 128 | 256 | unsplit |
|---|---|---|---|---|---|
| 100 | 20 (9-34) | 15 (13-24) | whole | whole | 17 (14-19) |
| 250 | 22 (10-33) | 16 (14-32) | 34 (30-42) | whole | 46 (42-51) |
| 521 | 34 (9-60) | 17 (11-31) | 26 (20-32) | 54 (46-64) | 232 (`and3` only) |

Warm (disk cache hot) bind of 250-521 rules, worst shape: 16 s at 64, 27 s at 128, 41 s at 256 and above (unsplit 250: 38 s).

`run()` over 200k rows, against parts of 32 (median): unsplit 1.4x slower at 100 rules, 1.8x at 250, 2.5x at 521 (`and3`); 128: 1.4x at 250, 1.5x at 521; 64: 1.1x. Peak RSS of the process was 500-740 MB at every size.

- Each extra part costs 15-20 us per `score()`. A request for a 250-rule tree is 800-950 us in every shape, so parts of 32 (8 parts) are the 12-13% the lead measured.
- Parts smaller than 64 never pay: they compile slower than 64 when the rules read different features (the `fraud` and 40-feature shapes took 50-75 s cold at 16, since every part has its own input types and so its own walker), and they are slower per request.
- At 521 rules `and3` runs 15-20% faster in parts of 64-128 than whole: a walker over hundreds of rules is slower per rule than a narrower one.
- Parts of 256 and unsplit trees compile too slowly: 46-64 s and 232 s cold for 521 rules.
- Cold compile time of 128 is bounded by the part, not the tree: 20-42 s from 250 to 521 rules.
- The other shapes behave like `and3`; the exceptions are the 40-feature shape (parts of 128 are 6% slower than unsplit at 250) and `isin_num` (6% in the repeat session).

## Decision
`TreeConfig.part_outputs` sets the part width: 128 by default (`_split.MAX_OUTPUTS`), `null` for an unsplit tree. At 128 a tree of up to 128 rules is not split, a 250-rule tree is within 3-6% of unsplit and cold bind of a 521-rule tree stays under 40 s (the bar was 60 s). The price is batch speed: `run()` over a big frame is 1.4-1.5x slower than with parts of 32, and the warm bind is slower than with parts of 64 (13-27 s against 5-16 s). A user who runs large frames sets `part_outputs: 32`; one who needs the last 3-6% and has a cold start to spare sets a larger one. See [configurable steps](../decisions/configurable-steps.md).
