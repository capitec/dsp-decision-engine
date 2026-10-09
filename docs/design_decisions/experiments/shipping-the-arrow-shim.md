# How should the first compiled artefact ship: wheels, compile on install, or a pure fallback?

Date: 2026-09-23 · Script: spike, in decider-experimentation/packaging-first-binary

## Purpose
Cost every step from "the Arrow shim is C" to a deployable install: build backend, wheel matrix, behaviour with no compiler or no wheel, vendored-source provenance, CI. Read it before replacing the build-on-first-import shim.

## Method
A miniature package (`d2shim`) laid out like the project, with the 6.7 KB shim and vendored nanoarrow 0.9.0 (159 KB), built four ways (setuptools, scikit-build-core, meson-python, a hatchling hook). Built, installed into throwaway venvs on CPython 3.14 and 3.12, 44 equivalence tests each. The Linux x86_64 leg ran in podman with cibuildwheel; the no-compiler and broken-binary cases were each simulated. Linux only: macOS, Windows (mingw cross-compile only) and aarch64 were not built.

## Result
- **Backend**: setuptools with `[[tool.setuptools.ext-modules]]` is 12 config lines plus an 88-line `PyInit` that exports each shim function's address as an int. It builds a `cp310-abi3` wheel, one per platform (five in all), 3 min 08 s for the Linux leg. scikit-build-core needs CMake; meson-python tagged the wheel `py3-none-any` with an x86_64 `.so` inside; a hatchling hook tagged `cp314` and so needs one wheel per Python.
- **glibc trap**: a wheel built on a current dev box linked `GLIBC_2.38`, and `auditwheel` refused `manylinux_2_28` (RHEL 8 is 2.28, RHEL 9 is 2.34). Only the manylinux container produced a deployable wheel (it needed `GLIBC_2.14` at most).
- **Packaging traps**: the sdist omitted the header until `depends=` named it; the wheel carried 335 KB of C source until `include-package-data = false`; a stale `build/` reused objects when flags changed; the module exported 89 symbols, 71 of them nanoarrow's, until `-fvisibility=hidden`.
- **No wheel**: an sdist install with no compiler fails; with a compiler but no `Python.h` (the Fedora system Python) it fails. `optional = true` turns that into a warning that scrolls past in a Docker log. Compile on install needs both, so it suits a developer machine, not a bank image.
- **Pure-numba fallback** (hand-decoded kernel): identical answers on 2.5M rows, 43.9 µs against 26.3 µs at one row, 1.15x at 1,000, 7% faster at 1M. It is the install story for musl, air-gapped and not-yet-supported CPythons, if the two are tested equal in every CI job and a strict environment switch makes a missing binary an error.
- **Provenance**: the pinned sha512 matches upstream's tarball and `bundle.py` regenerates the vendored files byte for byte. Vendored C is invisible to `pip-audit` and Dependabot, so it needs an SBOM entry and a security-advisory subscription. Estimated CI: about 25 runner-minutes a run, 6 wall.

## Decision
Not adopted. The code builds `shim.c` and nanoarrow with `$CC` on first import into `$XDG_CACHE_HOME/decider/arrow-shim`, under hatchling, with no pure fallback; a missing compiler raises one `ImportError`. A read-only home, a bank image without a compiler, or Windows needs the abi3 wheel above, built in a manylinux container, with `optional` left off. See [serving](../decisions/serving.md).
