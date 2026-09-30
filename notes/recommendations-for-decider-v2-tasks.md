Tasks:
# 1. Decimal types
ref: 2.1 of notes/recommendations-for-decider-v2.md
Design a Decimal type that can work in numba with all the arithmentic overloads we need a
Decimal(precision, scale)
it would be nice if it directly translated to a pyarrow decimal type
we need standard operations similar to the rawstring operations like a decider decimal operator set including 
round(decimal, precision=50) (not sure what is best here buit i suspect precision being an int is best)
```python
d: Decimal[10,2] = Decimal(204, scale=2, precision=10) # the same as 2.04
d2 = decider.ops.decimal.round(d, 5) # -> round to the nearest 5 so 204 -> 205 = 2.05
d3 = decider.ops.decimal.round(d1, 10) # -> round to the nearest 10 so 205 -> 210 = 2.1
d4 = decider.ops.decimal.round(d1, 10, method=HalfDown?) # Not sure on the best name but for edge cases how to handle it 205-> 200 = 2.0
```
Questions do we do 10 or 0.1 or 5 and 0.05 i would worry 0.05 falls under the same precision issues.
I think we can also make a standard guess on that all operations will generally deal with money so maybe be default to fit into a decimal64 we cater up to 99 trillion and 4 digits to handle cents so its a Decimal( scale=4, precision=18) as a default.

decimal probably needs a way to know the sign as well not sure if that is accounted for and we need to make sure rounding works correctly there. I think that means that round(x) = -round(-x).

do we also have Money default to a decimal type but Money[float] be explicit when you want to just represent money as a float in the flow. I think this is a good default to guide users to use the appropriate type.

Slightly adjacent task:
we have some string operations it would be good to structure the project to cater to a lot more in the future. Like numba compatible ops and types. ops.str and ops.decimal is ones we have now but i can see a lot more coming like lists and more similar to numpy/pandas operations.

# 2. A metadata tracing system
ref: 2.3
I think we need the ability to capture event traces something like
with engine.execution_context() as ctx:
    res = engine.run(payload)
ctx.trace()
ctx.events...
Ideally it would be great to see if its possible to use something like otel for this then every step could be its own span. maybe for the fused kernel we could even use the c or c# otel library to integrate with performance and its probably something that needs to be enabled like capture_trace=true for fused to allow speed vs observability

@step
def user_function(x: float, ctx: ExecutionContext): <- we inject this context in a. way that is numba safe
    ctx.tracer.log() # <- if the tracer is disabled or no context we inject a NoOp tracer. ideally this can be inlined 


