"""Scratch tests for decider2.compile.cache — doc 05 §4.1, §4.2.

EXPERIMENTS.md §C/§J2 are why every assertion here is about *identity*
(same digest -> same path -> same sys.modules name) rather than about numba
compile timing, which would make this suite slow and machine-dependent for
no extra coverage of the property that actually matters.
"""
from __future__ import annotations

import sys

from decider2.compile import cache


SOURCE_A = "value = 1\n"
SOURCE_B = "value = 2\n"


def test_content_hash_is_pure_and_stable():
    assert cache.content_hash(SOURCE_A) == cache.content_hash(SOURCE_A)
    assert cache.content_hash(SOURCE_A) != cache.content_hash(SOURCE_B)


def test_get_or_build_writes_a_real_file(tmp_path):
    cached = cache.get_or_build(SOURCE_A, tmp_path)
    assert cached.path.exists()
    assert cached.path.read_text(encoding="utf-8") == SOURCE_A
    assert cached.path.suffix == ".py"


def test_get_or_build_is_content_addressed_not_call_addressed(tmp_path):
    """Same bytes, called twice -> the same file and the same module name,
    whether or not this call is the one that wrote it."""
    first = cache.get_or_build(SOURCE_A, tmp_path)
    second = cache.get_or_build(SOURCE_A, tmp_path)
    assert first.path == second.path
    assert first.module_name == second.module_name
    assert first.module is second.module


def test_different_content_never_collides(tmp_path):
    a = cache.get_or_build(SOURCE_A, tmp_path)
    b = cache.get_or_build(SOURCE_B, tmp_path)
    assert a.path != b.path
    assert a.module_name != b.module_name
    assert a.module.value == 1
    assert b.module.value == 2


def test_a_content_only_edit_that_preserves_size_still_gets_a_new_file(tmp_path):
    """The doc 05 §4.2 / EXPERIMENTS.md §C failure mode: a same-slot literal
    edit that keeps file size (and, if mtime were normalised, mtime)
    identical must still be a cache MISS, never a stale HIT. Content
    addressing gets this for free because the file's *name* changes."""
    same_len_a = "value = 111\n"
    same_len_b = "value = 222\n"
    assert len(same_len_a) == len(same_len_b)
    a = cache.get_or_build(same_len_a, tmp_path)
    b = cache.get_or_build(same_len_b, tmp_path)
    assert a.path != b.path
    assert a.module.value == 111
    assert b.module.value == 222


def test_get_or_build_never_uses_spec_from_file_location(tmp_path):
    """EXPERIMENTS.md §J2: importing the generated driver by file location
    (rather than by module name) is the one thing that made the numba cache
    NOT survive a fresh process, even with byte-identical, mtime-preserved
    files. Assert the imported module is registered in sys.modules under
    the name cache.py derived, not merely importable by path."""
    cached = cache.get_or_build(SOURCE_A, tmp_path)
    assert sys.modules.get(cached.module_name) is cached.module


def test_list_cached_reports_every_written_file(tmp_path):
    cache.get_or_build(SOURCE_A, tmp_path)
    cache.get_or_build(SOURCE_B, tmp_path)
    names = cache.list_cached(tmp_path)
    assert len(names) == 2
    assert all(n.startswith("_gen_") and n.endswith(".py") for n in names)


def test_get_or_build_survives_a_simulated_restart(tmp_path):
    """doc 00-BUILD.md §2 ("Lifecycle"): a build-time cache entry must
    survive a runtime restart. Simulate "restart" by evicting the module
    from sys.modules (a fresh process would never have had it) and
    confirming a second get_or_build call still finds the file on disk and
    re-imports it under the identical name, without rewriting it."""
    first = cache.get_or_build(SOURCE_A, tmp_path)
    written_at = first.path.stat().st_mtime_ns
    del sys.modules[first.module_name]
    second = cache.get_or_build(SOURCE_A, tmp_path)
    assert second.module_name == first.module_name
    assert second.path.stat().st_mtime_ns == written_at  # never rewritten
    assert second.module.value == 1
