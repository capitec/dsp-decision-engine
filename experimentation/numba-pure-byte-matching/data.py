"""The two test columns. Realistic string lengths (13-22 bytes; the low
set is 15-18 bytes, the high set 17-20). One value in twelve is non-ASCII."""
import numpy as np
import polars as pl

WORDS = ["dog-savings-acct", "doge-credit-card", "cat-home-loan-x", "dogma-vehicle-fin",
         "hotdog-overdraft", "bird-fixed-deposit", "fish-notice-acct", "dogs-cheque-acct",
         "cow-flexi-savings", "dodo-tax-free-sav", "dig-global-one-x", "dögg-money-market"]
BASE = ["dog", "doge", "cat", "dogma", "hotdog", "bird", "fish", "dogs", "cow", "dodo", "dig", "dögg"]


def make_strings(n, cardinality, seed=7):
    rng = np.random.default_rng(seed)
    if cardinality == "low":
        return pl.Series("ft3", np.array(WORDS)[rng.integers(0, 12, n)])
    return pl.Series("ft3", [f"{BASE[i % 12]}-acct-{i:08d}" for i in range(n)])


def needles(cardinality, series):
    """(name, kind-name, needle, pcre2 regex) -- the same literal for every
    strategy, plus the regex PCRE2/polars would need for it."""
    import re
    exact = WORDS[0] if cardinality == "low" else series[42]
    return [
        ("prefix", "dog", "^dog"),
        ("suffix", "acct", "acct$"),
        ("substring", "dog", "dog"),
        ("substring_absent", "zzz", "zzz"),
        ("exact", exact, "^" + re.escape(exact) + "$"),
    ]
