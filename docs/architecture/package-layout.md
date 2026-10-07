# Package layout

!!! note "Draft"
    Outline only.

## Rules

- A leading underscore anywhere in a path means private; everything else is public
- Public packages are thin: an `__init__.py` over underscore-prefixed implementation files
- Packages import only downward through the layers (foundation, core, libraries, tooling)
- Only the core imports `_engine`

## Packages

| Layer | Packages |
|---|---|
| Foundation | `exceptions`, `ops`, `_registry`, `_serde` |
| Core | `steps`, `ir`, `trace`, `debug`, `config`, `_engine` |
| Libraries | `lab` (`data`, `checks`, `diff`, `testing`, `experiments`) |
| Tooling | `serving`, `_tooling` (`cli`, `mcp`, `bridge`) |

## To write

- One paragraph per package
- Extension points: the registry and entry-point groups
- How the layer rule is enforced
