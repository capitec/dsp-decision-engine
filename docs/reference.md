# API reference

The public API, generated from the docstrings and grouped by what you use each name for.

## Build steps

::: decikit.step

::: decikit.frame_step

::: decikit.flow

::: decikit.dag

::: decikit.branch

::: decikit.loop

::: decikit.each

::: decikit.optimise

::: decikit.EachMode

::: decikit.allow_fallback

::: decikit.helper

::: decikit.param

::: decikit.param_table

::: decikit.missing_as

::: decikit.ParamTable

::: decikit.load_flow

## Configurable steps

::: decikit.TreeConfig

::: decikit.DecisionTableConfig

::: decikit.ScorecardConfig

## Run

::: decikit.Engine

::: decikit.Executable

::: decikit.describe

## Extend

::: decikit.Step

::: decikit.ConfigurableStep

::: decikit.Value

::: decikit.ParamRef

::: decikit.StepRef

## Errors

::: decikit.DecikitError

::: decikit.FallbackWarning

::: decikit.HiddenStateWarning

::: decikit.ShadowedTableWarning

## Value types

::: decikit.ops.Text

::: decikit.ops.decimal.Currency

::: decikit.ops.decimal.Decimal

::: decikit.ops.Percent

::: decikit.ops.Duration

::: decikit.ops.Struct

::: decikit.ops.Columnar

::: decikit.ops.FieldMetadata

## Serving

::: decikit.serving.RequestHandler

## Config store

::: decikit.config.ConfigStore

::: decikit.config.JsonFileStore

::: decikit.config.Version

::: decikit.config.VersionPart

::: decikit.config.VersionedConfig

## Debug

::: decikit.debug.Session

::: decikit.debug.RunSummary

::: decikit.debug.Event

::: decikit.debug.Paused

::: decikit.debug.NodeStarted

::: decikit.debug.NodeFinished

::: decikit.debug.NodeVisited

::: decikit.debug.Overridden

::: decikit.debug.Edited

::: decikit.debug.ParamsValidated

::: decikit.debug.RunStarted

::: decikit.debug.RunFinished

::: decikit.debug.ReloadFailed

::: decikit.debug.WarningEvent

::: decikit.debug.ErrorEvent

## Trace

::: decikit.trace.TraceSink

::: decikit.trace.TraceEvent

::: decikit.trace.Kind

::: decikit.trace.Adapter

::: decikit.trace.PassThroughAdapter

::: decikit.trace.TraceLossError

## Lab: data

::: decikit.lab.data.read

::: decikit.lab.data.scan

::: decikit.lab.data.dataset

::: decikit.lab.data.datasets

::: decikit.lab.data.DatasetProvider

::: decikit.lab.data.FileProvider

::: decikit.lab.data.FolderProvider

::: decikit.lab.data.LoadedData

::: decikit.lab.data.InputFingerprint

::: decikit.lab.data.DatasetError

::: decikit.lab.data.LoadError

::: decikit.lab.data.RecordKeyError

::: decikit.lab.data.MissingKeyColumnError

::: decikit.lab.data.MissingRecordKeyError

::: decikit.lab.data.DuplicateRecordKeyError

## Lab: checks

::: decikit.lab.checks.run

::: decikit.lab.checks.standard

::: decikit.lab.checks.catalogue

::: decikit.lab.checks.check

::: decikit.lab.checks.Check

::: decikit.lab.checks.CheckContext

::: decikit.lab.checks.Finding

::: decikit.lab.checks.Severity

::: decikit.lab.checks.Report

::: decikit.lab.checks.SkippedCheck

::: decikit.lab.checks.Suppression

## Lab: diff and proof

::: decikit.lab.diff.structure

::: decikit.lab.diff.outputs

::: decikit.lab.diff.revisions

::: decikit.lab.diff.prove

::: decikit.lab.diff.Revisions

::: decikit.lab.diff.StructureDiff

::: decikit.lab.diff.OutputDiff

::: decikit.lab.diff.Proof

::: decikit.lab.diff.SmtEvidence

::: decikit.lab.diff.StepChange

::: decikit.lab.diff.StepMove

::: decikit.lab.diff.EdgeChange

::: decikit.lab.diff.ParamChange

::: decikit.lab.diff.ColumnDiff

## Lab: testing

::: decikit.lab.testing.assert_equivalent

::: decikit.lab.testing.corpus

::: decikit.lab.testing.no_recompile

## Lab: experiments

::: decikit.lab.experiments.Experiment

::: decikit.lab.experiments.ExperimentDef

::: decikit.lab.experiments.ExperimentResult

::: decikit.lab.experiments.Scenario

::: decikit.lab.experiments.ScenarioResult

::: decikit.lab.experiments.RunStatus

::: decikit.lab.experiments.run

::: decikit.lab.experiments.start

::: decikit.lab.experiments.validate

## Lab: references

::: decikit.lab.resolve_step

::: decikit.lab.source_location

::: decikit.lab.FlowDescription

::: decikit.lab.FlowRef

::: decikit.lab.StepRef

::: decikit.lab.EdgeRef

::: decikit.lab.ValueSlotRef

::: decikit.lab.RecordRef

## Trees and expressions

::: decikit.steps.trees.TreeConfig

::: decikit.steps.trees.expr_condition

::: decikit.steps.trees.Route

::: decikit.steps.trees.Visit

## Flow files

::: decikit.steps.flowfile.load_flow

::: decikit.steps.flowfile.read_flow

::: decikit.steps.flowfile.components
