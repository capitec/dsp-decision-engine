# Tree editor: validating an offline merge without the row lock

Date: 2026-10-11 · Branch: `ship/wa-merge-lock` · Scripts: `benchmarks/tree_editor_live_offline.py`,
`benchmarks/tree_editor_live_herd.py`

## Purpose
An offline batch replayed its base, checked conflicts and validated the merge (up to three `validation_errors` calls,
200 to 400 ms each on 521 rules) while holding the project's `SELECT ... FOR UPDATE`, so every other write to the
project waited. Can the validation move out of the lock without merging anything that was not validated as committed?

## Method
- Each attempt but the last reads one `REPEATABLE READ` snapshot without the lock, merges and validates there, then
  locks and commits only if the version has not moved. The third attempt holds the lock throughout, so a project that
  never goes quiet still merges. A conflict or validation refusal is decided on the snapshot (rows only accumulate, so
  a refusal there stands later).
- Batches of one project wait for each other per backend process (an `asyncio.Lock`), so they do not keep overtaking
  each other's attempts. Not a Postgres advisory lock: a waiter would hold a pool connection, and 30 waiters on one
  node would leave the holder none.
- Two-node stack (`stack.sh`, `E2E_NODES=2`, live on), generated-521 and fraud-live. Offline benchmark: 10 online
  renames, then a batch of 50 renames against them (n=10 per tree). Herd: an online client renames a rule every
  100 ms on node 1 while 10 offline batches of 50 arrive at once over both nodes (5 runs). Same stack and scripts
  before and after; lock time from the `live apply ... offline=True` log lines.

## Result
Offline benchmark (no contention; every batch commits on its first attempt):

| | merged round trip p50 / p95 | rejected p50 / p95 | row lock held p50 / p95 |
|---|---|---|---|
| before, generated-521 | 383 / 1063 ms | 104 / 123 ms | 360 / 1030 ms |
| before, fraud-live | 396 / 482 ms | 123 / 145 ms | (both trees) |
| after, generated-521 | 353 / 741 ms | 104 / 124 ms | 23 / 27 ms |
| after, fraud-live | 454 / 536 ms | 118 / 209 ms | (both trees) |

Herd (5 runs; per run: batch completion p50 and max, online round trips the herd overlapped):

| | batches done p50 | batches done max | online during p50 | online during max | online before p50 |
|---|---|---|---|---|---|
| before | 1845 to 2197 ms | 3363 to 3963 ms | 1716 to 3949 ms (n=1 or 2: blocked for the whole herd) | 3198 to 3949 ms | 68 to 80 ms |
| after, 3 attempts | 3911 to 4304 ms | 6500 to 7414 ms | 90 to 205 ms (n=21 to 24) | 472 to 581 ms | 63 to 70 ms |
| after, 2 attempts | 2525 to 2918 ms | 4272 to 4755 ms | 192 to 304 ms (n=10 to 13) | 479 to 708 ms | 68 to 74 ms |

- Without contention the lock is held 23 ms instead of 360; the round trip is unchanged (validation still runs).
- With an editor saving every 170 ms, no unlocked attempt survives (every herd batch took all three), so each batch
  still validates once under the lock (held 365 ms p50; validating took 335 ms p50 against about 250 ms alone, the
  rest waiting behind the node's own per-version validation in the same process). The online editor now waits for
  at most one such batch, not all ten; the batches take twice as long. Three attempts beat two for the online editor; two finish the herd sooner.
- Validation errors are now counted per (type, message): a merge is refused when it holds more copies than
  `max(theirs, mine, theirs + mine - base)`. `loc` is left out (rule positions shift in a merge) and so is `input`.
- The validator has a 30 s timeout; on a timeout (or a dead process) it is killed and a new one takes over. An
  offline batch it was checking is refused with a reason; the node's per-version validation resumes on the next
  version.

Open: a validator process of its own for merges would cut the locked fallback by about a quarter, for one more
process per node.
