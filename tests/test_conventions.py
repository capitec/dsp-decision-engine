import re
from pathlib import Path

DECIDER = Path(__file__).resolve().parent.parent / "decider"
MAX_LINES = 500
ALLOWLIST: dict[str, str] = {}  # {path relative to decider/: reason}
BANNED = re.compile(r"doc \d\d|§|EXPERIMENTS")


def _files():
    return sorted(DECIDER.rglob("*.py"))


def test_no_file_over_500_lines_unless_allowlisted():
    long = [
        f"{rel} ({n} lines)"
        for p in _files()
        if (rel := p.relative_to(DECIDER).as_posix()) not in ALLOWLIST
        and (n := len(p.read_text().splitlines())) > MAX_LINES
    ]
    assert not long, "split these into packages or allowlist them with a reason:\n" + "\n".join(long)


def test_no_references_to_docs_or_experiments():
    hits = [
        f"{p.relative_to(DECIDER)}:{i}: {line.strip()}"
        for p in _files()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if BANNED.search(line)
    ]
    assert not hits, "code must not reference docs, sections or experiments:\n" + "\n".join(hits)
