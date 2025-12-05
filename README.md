# North Pole

1. Do we want to use Hamilton to construct the pipeline or rather have a Hamilton element in the pipeline 

2. A close look at elements that are needed (decision trees, scorecards and decision tables) to check for differences. 

3. Ill try preparing an example of the merchant loan process so we can see what they need. 
https://github.com/capitecbankltd/dsp-team-bbc_re-unsecured-term-loan-credit-flow/blob/main/source_dir/utl_pricing/utl_pricing_properties.py
4. How are we going to do dumpoing of configs. ideally if we dump a scorecard:
Flow.add(Scorecard(
    1,5,2
))
should dump something like:
```json
{
    elements: [
        {
            op: "internal:Scorecard"
            lib: "north-polrs" #extension
            config: {1,5,2}
        }
    ]
}
```