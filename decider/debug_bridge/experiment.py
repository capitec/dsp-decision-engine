"""The bridge's `experiment` command: run an experiment headlessly and return its result as JSON."""
from pathlib import Path


def run_experiment(bridge, definition=None, def_=None, file=None, pipeline=None, data=None, out_dir=None):
    """Run an experiment from an `experiment.json` path (or its text), or an inline `def_` over `file`'s pipeline."""
    from decider.experiments import ExperimentDef, run, run_experiment
    if definition is not None:
        text = Path(definition).read_text() if not definition.lstrip().startswith("{") else definition
        result = run(ExperimentDef.model_validate_json(text), out_dir=out_dir)
    elif file is not None:
        bridge.describe(file, pipeline)
        result = run_experiment(ExperimentDef.model_validate(def_ or {}), bridge.step, bridge._rows(data),
                                out_dir=out_dir)
    else:
        raise ValueError("experiment needs 'definition' or 'file'")
    return result.model_dump(mode="json")
