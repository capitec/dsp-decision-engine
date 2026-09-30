"""The bridge's `experiment` command: run an experiment headlessly and return its result as JSON."""
from pathlib import Path


def run_experiment(bridge, yaml=None, def_=None, file=None, pipeline=None, data=None, out_dir=None):
    """Run an experiment from a YAML definition, or an inline `def_` over `file`'s pipeline."""
    from decider.experiments import ExperimentDef, run, run_experiment
    from decider.experiments.yaml import load as yaml_load
    if yaml is not None:
        text = Path(yaml).read_text() if "\n" not in yaml and Path(yaml).exists() else yaml
        result = run(ExperimentDef.model_validate(yaml_load(text)), out_dir=out_dir)
    elif file is not None:
        bridge.describe(file, pipeline)
        result = run_experiment(ExperimentDef.model_validate(def_ or {}), bridge.step, bridge._rows(data),
                                out_dir=out_dir)
    else:
        raise ValueError("experiment needs 'yaml' or 'file'")
    return result.model_dump(mode="json")
