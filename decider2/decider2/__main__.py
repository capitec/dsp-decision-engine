"""`python -m decider2 ...` — decider2 has no installed console script yet
(it is not in the repo's packaged wheel, see `pyproject.toml`'s
`packages = ["decider"]`), so this is today's `decider2 serve <pipeline.py>`
entry point."""
from decider2.cli import main

if __name__ == "__main__":
    main()
