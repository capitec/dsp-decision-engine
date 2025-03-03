# Backlog items
## Configuration Management
1. Automatic model Validation: We need a way to validate configuration as soon as its changed before deploying a model
2. Parameter changing: We need better ways to abstract certain config parameters to be changed by exec levels
    1. We potentially need a ui to enable quick parameter changing
    2. We need to flesh out standards for parameter stores (not only file stores)
    3. We need a way to enable validation on parameters so that they cannot be changed the erroneous values.
3. Seamless deployment: We need a way to enable deployments without users needing to be familiar with github
    1. We need this to integrate well with parameters (the process to update a value like vat from 14% to 15% should be a 1 min job not a 30 min job)
    2. We need a way to verify results perhaps with backtesting or live data before deploying
4. Integration with A/B testing: We must make sure that all the retraining and ab testing work that is done is compatible with spockflow
    1. Node-level: Users may want to select a subsection of the model to alter and do ab testing with so that they are able to see the outcomes midflow and how it affected the process as a whole (multiple ab tests at the same time)
5. Dashboarding/tracing monitoring
## Tree Functionality
1. We should make it easy to migrate from the old format to the new format
2. We need a way to optimise tree structures using libraries such as sympy
3. We need a way to make trees work better with different data types. Currently it only supports int and float values. We will need to extend it that if we want a categorical string we are able to draw out the compute of changing the value to categorical before.
4. We should consider bringing an easy way to split the data and rejoin it after a tree. This way we can cull off parts of the pipeline where there is no data to compute.
## Dashboarding
1. We should consider creating easy to use dashboarding functionalities into the ui so that we can monitor the outputs of the flows.