"""fraud_interdiction -- transaction fraud and scam interdiction (spec 01).

A real-time flat rule set (~520 live, ~100 shadow rules across 5 families)
evaluated against every instant payment, with a separately-governed
overlay stack, action precedence over the firing set, and a decision
record meeting the 09 §5.15 evidence contract. See NOTES.md.
"""
