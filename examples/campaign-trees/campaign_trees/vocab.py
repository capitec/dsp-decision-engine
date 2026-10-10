"""Codes the assignment record uses."""

# The tree's own verdict is DO_NOT_TARGET or TARGET; the assignment's can also be the two non-contact outcomes.
DO_NOT_TARGET = 0
TARGET = 1
HOLD_OUT = 2  # the tree says target, but the client is in a control group or the universal holdout
SUPPRESSED = 3  # the tree says target, but a measurement-relevant suppression forbids the contact
