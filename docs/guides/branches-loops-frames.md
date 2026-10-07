# Branches, loops and frame steps

!!! note "Draft"
    Outline only.

Different logic per row, iteration, and whole-frame work.

- `branch`: two arms, and one arm that passes other rows through
- `loop`: carried values and `max_iterations`
- `@frame_step`: joins, group-bys, ragged lists
- Per-item rules with `each`; best-of-N with `optimise`
- Strings (`Text`) and nested columns (`Struct`, `Columnar`)
- What compiles and what falls back (`@allow_fallback`)
