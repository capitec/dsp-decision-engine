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

**Packed branches and loops** (fused mode, `packed-control-flow.md`) do have
control flow: an arm or a loop body may run on no row. With lazy validation, a
branch or loop packs only when no call in it but its own condition (which runs
whenever the branch or loop does) has params; otherwise it runs the
Python-driven path, which validates each call as it runs. Eager validation has
checked every node before any kernel launches, so it packs regardless. This is
decided once per executable, so a session's checkpoints don't depend on the
params document.

**Revisit when** that costs too much (a lazy pipeline whose hot branch has
params in its arms) or a tree's internal nodes carry params: then
suspend/resume inside the kernel (status array in, `(node, row)` out, relaunch
from the row) keeps "only nodes that run can fail" while packing everything.

**One difference:** when one node of a kernel has invalid params and an
earlier node of the same kernel would raise a runtime error, fused mode reports
the params error first; interpreted mode reports the runtime error.
