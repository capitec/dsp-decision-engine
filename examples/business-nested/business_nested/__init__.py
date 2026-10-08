"""business_nested: a business credit decision over a nested structure of entities and their adverse events.

`business_nested.pipeline.build()` is the whole decision. The pieces it is made of are public modules:
`structure`, `events`, `rollup`, `scoring`, `entity`, `blend`, `financial`, `grade`, `pricing`,
`sole_proprietor`, `outcome` and `counterfactual`. See the README for what each publishes.
"""
