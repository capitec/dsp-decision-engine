"""The bridge's experiment commands: run headlessly with live progress, build schemas, and validate."""
from pathlib import Path

from decider.contract import JobHandle


def _load_definition(bridge, definition, def_, file, pipeline):
    """An `ExperimentDef` from `definition` (a path or its JSON text) or an inline `def_` over `file`'s pipeline."""
    from decider.experiments import ExperimentDef
    if definition is not None:
        text = Path(definition).read_text() if not definition.lstrip().startswith("{") else definition
        return ExperimentDef.model_validate_json(text)
    if file is not None:
        bridge.describe(file, pipeline)
        return ExperimentDef.model_validate(def_ or {})
    raise ValueError("needs 'definition' or 'file'")


def run_experiment(bridge, req, definition=None, def_=None, file=None, pipeline=None, data=None, out_dir=None,
                   input=None, dataset=None):
    """Run an experiment from an `experiment.json` path (or its text), or an inline `def_` over `file`'s pipeline.

    `input` (a path) or `dataset` (a project dataset name), when given, override the definition's own input
    (running the same experiment on other data). Streams `{"progress": {"done", "total"}}` lines on `req`'s id while it runs; a `{"cmd": "cancel",
    "target": <that id>}` line stops it at the next batch boundary.
    """
    from decider.experiments import InputSpec, run, run_experiment
    from decider.experiments.callbacks import Callback

    class _StreamProgress(Callback):
        def on_progress(self, run_, done):
            bridge._reply(req, {"progress": {"done": done, "total": run_.total}})

    d = _load_definition(bridge, definition, def_, file, pipeline)
    if input is not None:
        d.input = InputSpec(path=input)
    elif dataset is not None:
        d.input = InputSpec(dataset=dataset)
    job = JobHandle(job_id=str(req["id"]), kind="experiment_run")
    bridge._jobs[req["id"]] = job
    try:
        if definition is not None:
            result = run(d, job=job, callbacks=[_StreamProgress()], out_dir=out_dir)
        else:
            result = run_experiment(d, bridge.step, bridge._rows(data), job=job, callbacks=[_StreamProgress()],
                                    out_dir=out_dir)
    finally:
        bridge._jobs.pop(req["id"], None)
    return result.model_dump(mode="json")


def experiment_schema(bridge):
    """`{metrics, plots, blocks}` JSON Schemas for every registered type, for building a UI form.

    Each metric/plot schema also carries the `Summary` type it gives or needs (`summary`/`requires`, by class
    name), so a UI can offer only the plot types that fit a chosen metric.
    """
    from decider.experiments import Block, Metric, Plot

    def annotate(root, attr):
        schemas = root.json_schema()
        for tag, schema in schemas.items():
            schema[attr] = getattr(root.resolve(tag), attr).__name__
        return schemas

    return {"metrics": annotate(Metric, "summary"), "plots": annotate(Plot, "requires"), "blocks": Block.json_schema()}


def experiment_validate(bridge, definition=None, def_=None, file=None, pipeline=None):
    """Every problem in the definition against its flow (unknown steps, bad overrides, ...), without running it."""
    from decider.experiments import validate
    from decider.experiments.runner import load_flow

    d = _load_definition(bridge, definition, def_, file, pipeline)
    step = load_flow(d.flow.entry) if definition is not None else bridge.step
    return {"problems": validate(d, step)}
