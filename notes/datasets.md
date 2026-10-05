# Datasets and providers

The VS Code side panel has a Datasets view, backed by a `DatasetProvider`
registry in `decider/data/providers/`.

## Why

- Main has no dataset layer to port. Its `_ext.py` (`create_extendable_model`,
  `register_provider`, a `decider_extensions/` folder imported for side
  effects) is the pattern `BaseRegistryModule` already replaces: lookup by
  import path, lazy aliases, `json_schema()` for forms, no `model_rebuild` per
  registration.
- Today two readers exist: `decider.data.load` (eager) and
  `experiments.runner.scan` (lazy). Experiments and debug runs name data by raw
  path, so nothing remote can be referenced portably.
- Neither extension has a datasets view. The new one has a `Sample` /
  `Choose a file…` quick-pick in `src/debugging.ts`.

## Decisions

- **A provider is one dataset.** Not a catalogue: the catalogue is just
  `name -> provider`.
- **Clients bring their own sources.** Redshift, Athena and the like are client
  classes named by import path in `config.json`. They read their own
  credentials (usually the default AWS chain). Decider does no `${VAR}`
  substitution.
- **Built-ins are `file` and `folder` only.** `s3://` paths work through polars
  scans with default credentials, so there is no `s3` type.
- **A folder dataset has one format.** The format is inferred from its parts,
  and mixed formats are an error.
- **`key` declares the record key.** When set, it replaces the id-column guess,
  `check_key` validates it on load, and it becomes the durable `RecordRef` key.
  Content hashing stays the default fingerprint. A provider may override
  `fingerprint()` with something cheaper (table version, ETag).
- **Read-only for now.** Providers have no `save`. A `save(frame, mode)` with
  replace or append was considered as an escape hatch for code outside the
  panel. It was left out until something needs it: the panel would never offer
  it, and the write modes add surface to every provider.
- **Explicit wins.** A `config.json` entry with the same name as an inferred
  dataset replaces it. That is how a local folder gets a `key`.
- **Python is the only source of truth.** The extension asks the bridge and has
  no datasets setting of its own.

## Interface

```python
class DatasetProvider(BaseRegistryModule, root=True):
    model_config = ConfigDict(extra="forbid")

    key: tuple[str, ...] | None = None

    @abstractmethod
    def scan(self) -> pl.LazyFrame: ...

    def schema(self) -> dict[str, str]: ...          # scan().collect_schema()
    def head(self, n: int = 50) -> pl.DataFrame: ...
    def fingerprint(self, frame: pl.DataFrame) -> str | None: ...   # row hash
    def load(self) -> LoadedData: ...                # collect, fingerprint, key
```

| type | fields | `scan` |
|---|---|---|
| `file` | `path`, `format=None` | `scan_parquet` / `scan_csv` / `scan_ndjson` by suffix |
| `folder` | `path`, `format=None` | Glob over `{path}/*.{ext}`; Parquet uses `hive_partitioning`. |

## Resolution

`DeciderSettings.datasets_path = "${PROJECT_ROOT}/datasets"` (env
`Decider_DATASETS_PATH`). `${PROJECT_ROOT}` is the only placeholder: the
nearest ancestor holding `configs/` (`debug_bridge.loading.project_root`), or
the cwd when there is none. Any other relative path, such as `"./datasets"`,
is relative to the cwd. The extension launches Python with the cwd set to the
flow file's directory, which is why the default goes through the project root.
`config.json` entries get no placeholder expansion.

`datasets(start=None, path=None) -> dict[str, DatasetProvider]` validates
every entry and raises one `DatasetError` naming all the bad ones.
`dataset(name, start=None, path=None)` validates only that entry, so a broken
sibling never stops a run; experiments and the debug bridge use it.
`dataset_documents()` is the unvalidated `{name: document}` both build on, and
`datasets_folder()` the resolved folder. The entries:

1. `{path}/{name}.{csv,parquet,jsonl,ndjson,json}` becomes
   `FileProvider(path=...)`.
2. `{path}/{name}/` becomes `FolderProvider(path=...)`.
3. Each `{path}/config.json` entry `{name: provider document}` is validated
   through `DatasetProvider.resolve(type)` and replaces any inferred dataset of the same
   name. Relative paths in an entry resolve against the datasets folder.

```json
{
    "applications": {"type": "mypkg.data:RedshiftTable", "table": "risk.applications", "key": ["application_id"]},
    "bureau":       {"type": "mypkg.data:AthenaQuery", "query": "select * from bureau.latest"},
    "loans":        {"type": "folder", "path": "loans", "key": ["client_id"]}
}
```

Listing never opens data. The bridge's `datasets` command reports a bad entry
per name, so one bad entry doesn't hide the rest.

## Consolidation

- `data/formats.py` holds the one suffix-to-reader table, used by `file`,
  `folder`, `load()` and experiments. `load(path)` becomes `scan().collect()`
  plus the fingerprint. Bytes and frame sources stay as they are.
  `experiments.runner.scan` is deleted; `decider.experiments.scan` re-exports
  the formats one because it is public.
- `InputSpec` gains `dataset: str` next to `path`. That name is what goes into
  `InputFingerprint.dataset` and `RecordRef.dataset`.

## Bridge and extension

Bridge commands (short-lived `ask()`, like `describe`):

| cmd | args | result |
|---|---|---|
| `datasets` | `file?` | `{root, datasets: [{name, type, key, path, error?}]}`; `path` only for local file/folder datasets |
| `dataset_schema` | `name`, `file?` | `{columns: [[name, dtype]], row_count}`; row count only from Parquet metadata, else `null` |
| `dataset_head` | `name`, `n`, `file?` | `{columns, rows}` |
| `experiment` | adds `dataset` | runs the definition on that dataset |

`file` (a flow file) locates the project; without it the cwd does. `start`,
`trace` and `sweep` take `dataset=` beside `data=`.

`run.py` gains `--dataset <name>` beside `--data <path>`, so a string is never
guessed to be one or the other.

The `decider.datasets` view:

- Each dataset expands to its columns, with their types.
- Actions: Preview (click), Debug with… (inline), Copy name, Reveal (file and
  folder only), Refresh. No "Use in experiment": the extension has no
  experiment command yet.
- The view follows the shown flow: it lists the datasets of that flow's
  project (the nearest ancestor holding `configs/`) and names the project in
  the view's title. With no flow shown, or one outside any project, it lists
  every project in the workspace, one node each when there are several. "Debug
  with…" offers only flows inside the dataset's project, because a name
  resolves from the flow's project. Code outside any project (the loose
  examples) has no stable datasets folder: it depends on the cwd.
- A file watcher on the datasets folder refreshes the view.
- The debug quick-pick lists datasets above `Choose a file…`.

## Template and `.env`

The starter template is unchanged: the default already points at
`${PROJECT_ROOT}/datasets`. Writing `${PROJECT_ROOT}` in a `.env` doesn't work:
`decider serve` loads `.env` with python-dotenv 1.2, which expands `${...}`
even inside single quotes, so an unset `PROJECT_ROOT` becomes empty
(`/datasets`). `GUIDE.md` says to use an absolute or cwd-relative path there.
