# Lazy params validation in fused mode: validate a kernel's nodes as it launches

**Decision:** In fused (and stepped) mode with `params_validation="lazy"`,
the Python driver validates every node of a kernel right before it launches
the kernel. There is no suspend/resume inside the kernel, and no up-front
validation of the whole document either.

**Why:** A fused kernel is one innermost sequence of scalar steps, with no
control flow between them: branches and loops are driven in Python, and their
conditions, arms and bodies are separate kernels. So every node of a kernel
runs on every row the kernel runs on. A kernel that suspends at the first
`UNKNOWN` node would always suspend at row 0 of that node, validate it and
relaunch at row 0; that is the same set of validations, in the same order, as
validating all the kernel's nodes before launching it, with a kernel
re-entry per node and a status-array argument added for nothing.

Behaviour is what lazy promises: only nodes that run can fail, an invalid param
in an arm no row takes never fails, and `report.validated` lists exactly the
nodes that ran (checked in stepped and fused by
`tests/run/test_params_validation.py` and
`tests/run/test_compiled_modes.py`).

**Revisit when** a kernel gains control flow (a fused branch, or a tree whose
internal nodes carry params): then a node inside the kernel may run on no row,
and suspend/resume (status array in, `(node, row)` out, relaunch from the row)
is the way to keep "only nodes that run can fail".

**One difference:** when one node of a kernel has invalid params and an
earlier node of the same kernel would raise a runtime error, fused mode reports
the params error first; interpreted mode reports the runtime error.
