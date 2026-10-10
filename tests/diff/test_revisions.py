"""revisions: two git revisions of one entry point, each loaded in its own process."""
import polars as pl
import pytest

from decikit.lab import diff
from decikit.exceptions import DecikitError

ENTRY = "shop.pipeline:build"
FRAME = pl.DataFrame({"income": [1000.0, 500.0], "debt": [350.0, 400.0]})


def test_revisions_report_which_rows_changed(repo):
    o = diff.revisions("HEAD^", "HEAD", entry=ENTRY).outputs(FRAME)
    assert o.rows_changed == 2 and o.columns["ratio"].changed == 2


def test_revisions_report_what_changed_structurally(repo):
    s = diff.revisions("HEAD^", "HEAD", entry=ENTRY).structure()
    assert not s.identical
    assert [c.ref.path for c in s.changed_steps] == ["credit/ratio"]


def test_the_same_revision_twice_is_identical(repo):
    r = diff.revisions("HEAD", "HEAD", entry=ENTRY)
    assert r.structure().identical and r.outputs(FRAME).rows_changed == 0


def test_an_unknown_revision_is_an_error(repo):
    with pytest.raises(DecikitError, match="did not resolve"):
        diff.revisions("nope", "HEAD", entry=ENTRY).structure()


def test_resolve_gives_none_for_an_unknown_revision_and_a_sha_for_a_known_one(repo):
    from decikit.lab.diff._revisions import resolve

    assert resolve("nope") is None
    assert len(resolve("HEAD")) == 40
