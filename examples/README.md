# Examples

Worked projects built on `decider`. Each directory is a project you can `cd` into, build and serve; its
`README.md` says what it decides and which `decider` features it shows. Later projects reuse earlier ones
by package name (`credit_core` from `shared-credit-core`, and so on).

Put every project on the path once, then work from any project directory:

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/<project>
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="<package>.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
uv run decider build && uv run decider serve
```

`uv run pytest tests/examples` runs every project's own tests, one process per project.
