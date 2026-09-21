

def register_credit_modules():
    from .decision_table.module import DecisionTableModule
    from .scorecard.module import (
        ScoreCard,
        ProbabilityDefault,
        LogProbability,
        ScoreFromPDO,
        MergeScorecardValues,
    )
    from decider.modules import register_graph_module

    MODULE_LIST = [
        DecisionTableModule,
        ScoreCard,
        ProbabilityDefault,
        LogProbability,
        ScoreFromPDO,
        MergeScorecardValues,
    ]

    for module_cls in MODULE_LIST:
        register_graph_module(module_cls)