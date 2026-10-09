# API reference

The public API, generated from the docstrings and grouped by what you use each name for.

## Build steps

::: decider.step

::: decider.frame_step

::: decider.flow

::: decider.dag

::: decider.branch

::: decider.loop

::: decider.each

::: decider.optimise

::: decider.EachMode

::: decider.allow_fallback

::: decider.helper

::: decider.param

::: decider.param_table

::: decider.missing_as

::: decider.ParamTable

::: decider.load_flow

## Configurable steps

::: decider.TreeConfig

::: decider.DecisionTableConfig

::: decider.ScorecardConfig

## Run

::: decider.Engine

::: decider.Executable

::: decider.describe

## Extend

::: decider.Step

::: decider.ConfigurableStep

::: decider.Value

::: decider.ParamRef

::: decider.StepRef

## Errors

::: decider.DeciderError

::: decider.FallbackWarning

::: decider.HiddenStateWarning

## Value types

::: decider.ops.Text

::: decider.ops.decimal.Currency

::: decider.ops.decimal.Decimal

::: decider.ops.Percent

::: decider.ops.Duration

::: decider.ops.Struct

::: decider.ops.Columnar

::: decider.ops.FieldMetadata

## Serving

::: decider.serving.RequestHandler

## Config store

::: decider.config.ConfigStore

::: decider.config.JsonFileStore

::: decider.config.Version

::: decider.config.VersionPart

::: decider.config.VersionedConfig

## Debug

::: decider.debug.Session

::: decider.debug.RunSummary

::: decider.debug.Event

::: decider.debug.Paused

::: decider.debug.NodeStarted

::: decider.debug.NodeFinished

::: decider.debug.NodeVisited

::: decider.debug.Overridden

::: decider.debug.Edited

::: decider.debug.ParamsValidated

::: decider.debug.RunStarted

::: decider.debug.RunFinished

::: decider.debug.ReloadFailed

::: decider.debug.WarningEvent

::: decider.debug.ErrorEvent

## Trace

::: decider.trace.TraceSink

::: decider.trace.TraceEvent

::: decider.trace.Kind

::: decider.trace.Adapter

::: decider.trace.PassThroughAdapter

::: decider.trace.TraceLossError

## Lab: data

::: decider.lab.data.read

::: decider.lab.data.scan

::: decider.lab.data.dataset

::: decider.lab.data.datasets

::: decider.lab.data.DatasetProvider

::: decider.lab.data.FileProvider

::: decider.lab.data.FolderProvider

::: decider.lab.data.LoadedData

::: decider.lab.data.InputFingerprint

::: decider.lab.data.DatasetError

::: decider.lab.data.LoadError

::: decider.lab.data.RecordKeyError

::: decider.lab.data.MissingKeyColumnError

::: decider.lab.data.MissingRecordKeyError

::: decider.lab.data.DuplicateRecordKeyError

## Lab: checks

::: decider.lab.checks.run

::: decider.lab.checks.standard

::: decider.lab.checks.catalogue

::: decider.lab.checks.check

::: decider.lab.checks.Check

::: decider.lab.checks.CheckContext

::: decider.lab.checks.Finding

::: decider.lab.checks.Severity

::: decider.lab.checks.Report

::: decider.lab.checks.SkippedCheck

::: decider.lab.checks.Suppression

## Lab: diff and proof

::: decider.lab.diff.structure

::: decider.lab.diff.outputs

::: decider.lab.diff.revisions

::: decider.lab.diff.prove

::: decider.lab.diff.Revisions

::: decider.lab.diff.StructureDiff

::: decider.lab.diff.OutputDiff

::: decider.lab.diff.Proof

::: decider.lab.diff.SmtEvidence

::: decider.lab.diff.StepChange

::: decider.lab.diff.StepMove

::: decider.lab.diff.EdgeChange

::: decider.lab.diff.ParamChange

::: decider.lab.diff.ColumnDiff

## Lab: testing

::: decider.lab.testing.assert_equivalent

::: decider.lab.testing.corpus

::: decider.lab.testing.no_recompile

## Lab: experiments

::: decider.lab.experiments.Experiment

::: decider.lab.experiments.ExperimentDef

::: decider.lab.experiments.ExperimentResult

::: decider.lab.experiments.Scenario

::: decider.lab.experiments.ScenarioResult

::: decider.lab.experiments.RunStatus

::: decider.lab.experiments.run

::: decider.lab.experiments.start

::: decider.lab.experiments.validate

## Lab: references

::: decider.lab.resolve_step

::: decider.lab.source_location

::: decider.lab.FlowDescription

::: decider.lab.FlowRef

::: decider.lab.StepRef

::: decider.lab.EdgeRef

::: decider.lab.ValueSlotRef

::: decider.lab.RecordRef

## Trees and expressions

::: decider.steps.trees.TreeConfig

::: decider.steps.trees.expr_condition

::: decider.steps.trees.Route

::: decider.steps.trees.Visit

## Flow files

::: decider.steps.flowfile.load_flow

::: decider.steps.flowfile.read_flow

::: decider.steps.flowfile.components
