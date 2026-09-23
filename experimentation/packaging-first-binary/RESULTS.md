# Shipping decider2's first binary: the packaging and build story, costed

*For a reader with no prior context. decider2 is a numba-first decision
engine that today installs as pure Python (`pip install`, no compiler). The
owner has decided to adopt nanoarrow — Apache's single-file C library for the
Arrow format — so that a numba kernel can read polars string columns through
nanoarrow's accessors instead of hand-decoded layout code. That needs a small
C shim (`experimentation/nanoarrow-accessors/c/nashim.c`, 6.7 KB, plus the
vendored `nanoarrow.c` amalgamation, 159 KB). The 1.6-second gcc build was
already measured; this note costs everything that follows from shipping that
object — build backend, wheel matrix, the no-compiler fallback, install-time
behaviour, supply chain, CI — and proves each claim with a prototype under
`experimentation/packaging-first-binary/` that was actually built, installed
into throwaway venvs, and tested. Linux x86_64 only; §8 lists exactly what
this box could not test.*

---

## The recommendation

**Build backend: stay on setuptools.** decider2 already uses
`setuptools.build_meta`. Compiling the shim is 12 declarative lines in
`pyproject.toml` (`[[tool.setuptools.ext-modules]]`, setuptools ≥ 74.1) plus
an 88-line C file that turns the shim into a real CPython extension module
built against the stable ABI (`Py_LIMITED_API` 3.10). **No `setup.py`, no
new build dependency, one wheel per platform** (`cp310-abi3-<platform>`, five
wheels in total) that every CPython ≥ 3.10 can use — proven here by
installing the same wheel into a 3.14 and a 3.12 venv, 44 tests passing in
both. The alternatives were also built (§2): scikit-build-core is the only
one that does this job as well, and it costs a second build system
(CMake) for no gain at one C file; meson-python and a hatchling hook each
have a wheel-tagging trap that the prototype hit.

**Keep the pure-numba kernel, as a policy-controlled fallback, not a silent
one.** Measured in the same process on the same data (§4): identical answers
on 2.5 M rows across single-chunk, two-chunk and sliced columns; the compiled
path is 1.67× faster at batch size 1 (26.3 µs vs 43.9 µs — a 17 µs
difference on a 353 µs single-record path, 5 %), and 7 % *slower* at a
million rows. The fallback is what every platform outside the wheel matrix
runs (musl/Alpine, an air-gapped RHEL box with no compiler, a future CPython
before the wheels catch up), it already exists with its own 38 tests, and the
compiled kernel shares its tree-walk and byte-matcher verbatim; the only
duplicated logic is "where does row *i*'s bytes start" — 9 lines of nanoarrow
call vs 13 lines of hand decode. Two implementations become a trap only if
they are allowed to drift; the guard is the equivalence suite in
`proto/tests/` running on both in every CI job, and the default policy
(§5): auto-fallback with one `RuntimeWarning`, `DECIDER2_STRINGS=compiled`
to make a missing binary a hard error in production images.

**The single biggest operational risk: a wheel that does not load on the
bank's servers, and nothing telling you.** Two facts from the prototype
combine. (1) A wheel built on this developer box links `GLIBC_2.38` and
`auditwheel` *refuses* to retag it `manylinux_2_28` (RHEL 8 is glibc 2.28,
RHEL 9 is 2.34): the only way to get a deployable Linux wheel is to build
inside the `quay.io/pypa/manylinux_2_28` container via cibuildwheel — the
release path becomes a five-runner binary pipeline the project has never
operated, of which the macOS and Windows legs cannot be verified from here.
(2) When the binary is missing or unloadable, the package still imports and
still gives right answers — through the fallback, 1.7× slower on the
single-record path — so a broken wheel in a production image is invisible
unless the strict policy is set. The mitigation is in the prototype and
costs nothing: `DECIDER2_STRINGS=compiled` in every production Dockerfile,
`decider2 doctor` (the `diagnose()` output in §5) in the smoke test, and
the weekly CI run so toolchain drift is caught before a release needs it.

---

## 1. What was built, and how to reproduce it

```
experimentation/packaging-first-binary/
  RESULTS.md                    this file
  build_and_test.sh             sdist+wheel with setuptools -> auditwheel -> install on 3.14 and 3.12 -> pytest    (~60 s)
  run_cibuildwheel.sh           the Linux x86_64 leg of the matrix, exactly as CI would, in podman                (~3.5 min)
  simulate_missing.sh           the no-wheel cases: no compiler, no headers, optional ext, broken .so            (~30 s)
  bench_fallback.py             pure vs compiled: identical answers, then µs per call at 1 / 100 / 1k / 1M rows (~3 min)
  provenance_check.sh           re-derive the vendored nanoarrow from the upstream tarball, byte for byte
  mingw_check.sh                cross-compile the shim for Windows with mingw-w64 in a container (partial Windows evidence)
  ci/wheels.yml                 the proposed GitHub Actions workflow (NOT installed in .github/)
  alt-backends/                 the three other backends, each a buildable miniature package + run_alts.sh
  proto/                        the miniature package `d2shim`, laid out like decider2 (src/, setuptools)
    pyproject.toml                the ONLY config: decider2's layout + 3 blocks marked "+ binary"
    src/d2shim/c/nashim.c         byte-identical copy of the experiment's shim (sha1 c6dfe9ee...)
    src/d2shim/c/nashim_module.c  NEW, 88 lines: PyInit__nashim exporting each sm_* address as an int (stable ABI)
    src/d2shim/vendor/            nanoarrow 0.9.0 amalgamation, LICENSE, NOTICE, VERSION (as vendored) + SHA512SUMS (new)
    src/d2shim/compiled.py        nashim.py from the experiment, addresses from _nashim instead of ctypes.CDLL
    src/d2shim/kernel_compiled.py kernel_b.py from the experiment (imports only)
    src/d2shim/pure.py            arrow-strings-in-tree/kernel.py, unchanged
    src/d2shim/_arrowc.py         arrow-strings-in-tree/arrowc.py, unchanged
    src/d2shim/strings.py         NEW, 108 lines: backend policy, run_tree() facade, diagnose()
    tests/test_equivalence.py     44 tests: both backends vs a Python reference; the policy, each case in a subprocess
  .tmp/                         throwaway venvs, wheels, logs (gitignored; every log quoted below is here)
```

Reproduce: `./build_and_test.sh`, then `./run_cibuildwheel.sh` (needs podman
and ~1.6 GB for the manylinux image), then `./simulate_missing.sh`,
`.tmp/venv314/bin/python bench_fallback.py`, `./provenance_check.sh`.
Everything runs in venvs under `.tmp/` on uv-managed CPython 3.14.5 and
3.12.12 (`uv python find 3.14 --managed-python`); the project's `.venv` is not
touched. Note the managed interpreter is required: **the system
`/usr/bin/python3.14` on this Fedora box has no `Python.h`** (no
`python3-devel`), which is itself a finding (§5, case A2).

The decider2 source tree was not modified. The shim and kernels were copied
from `experimentation/nanoarrow-accessors/` and
`experimentation/arrow-strings-in-tree/` in the main checkout (they are on
`feature/decider-v2`, commit af52b8d; this worktree branched earlier and does
not contain them).

---

## 2. Build backend: four options, all four built

Each was given the same job — compile `nashim.c` + `nanoarrow.c` into a
wheel that imports and reports `nanoarrow 0.9.0` — as a miniature package,
built with `uv build` in an isolated PEP 517 environment on CPython 3.14.
Dependency cost is what a fresh venv gets from `uv pip install <backend>`
(`.tmp/depcost.log`). Config lines are non-blank, non-comment lines of
`pyproject.toml` plus the backend's own build file.

| | **setuptools** (recommended, `proto/`) | scikit-build-core (`alt-backends/skbuild`) | meson-python (`alt-backends/meson`) | hatchling custom hook (`alt-backends/hatch`) |
|---|---|---|---|---|
| already decider2's backend | **yes** | no | no | no (decider v1 uses it) |
| build-time packages, MB | setuptools: **1 pkg, 4 MB** (already required) | 3 pkgs, 3 MB **+ CMake ≥ 3.15 and ninja** (system, or fetched as PyPI wheels: `cmake` ~30 MB, `ninja` ~0.3 MB) | 4 pkgs, 6 MB (meson) **+ ninja** | 6 pkgs, 2 MB; **a C compiler invocation you write yourself** |
| config added for the binary | **12 lines** `[[tool.setuptools.ext-modules]]` + 2 `[tool.distutils.bdist_wheel]` (+ 88-line `nashim_module.c`) | 19 lines (pyproject 8 + `CMakeLists.txt` 7 + excludes) | 16 lines (pyproject 6 + `meson.build` 10) | 28 lines (pyproject 12 + `hatch_build.py` 16) |
| what the artefact is | a real extension module `_nashim.abi3.so`; sm_* addresses read from it | plain `libnashim.so` loaded with `ctypes.CDLL` | plain `libnashim.so` | plain `libnashim.so` |
| wheel tag produced | **`cp310-abi3-linux_x86_64`** — one wheel per platform, CPython ≥ 3.10 | **`py3-none-linux_x86_64`** — one wheel per platform, any Python | first attempt **`py3-none-any`** (!): a pure-Python tag with an x86_64 `.so` inside, because `py.get_install_dir()` defaults to purelib; with `pure: false` → `cp314-cp314-linux_x86_64`, i.e. **one wheel per Python version** unless retagged | `cp314-cp314-linux_x86_64` (`infer_tag`): one wheel per Python version unless retagged with `wheel tags` |
| needs `Python.h` at build | yes (the PyInit file) | no | no | no |
| Windows exports | **handled**: the linker exports `PyInit__nashim`; addresses travel as ints, so nothing else needs exporting | `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON` (1 line) | needs a `.def` file or `__declspec` in `nashim.c` | not handled; hook is Unix-only as written |
| build time (3 C files, this box) | 4.9 s incl. sdist, isolated env | 3.2 s | 4.4 s | 2.2 s |
| `.so` in wheel | 301 KB unstripped (Python's `-g -O3` CFLAGS), 1 exported symbol | 76 KB, `-O2`, all symbols exported | 84 KB | 76 KB |
| editable install (`uv pip install -e`) | standard | supported (scikit-build-core rebuilds on import, opt-in) | supported via an import hook | standard, but the hook runs on every build |
| cibuildwheel / auditwheel / abi3audit understand it | **yes, natively** (the cibuildwheel run in §3 includes an `abi3audit --strict` pass) | yes | yes | yes, after retag |
| verdict | **adopt** | the right tool if the C grows to a real library with tests of its own; overkill for one file | the tagging default is a trap for exactly this use; skip | a hand-rolled compiler call with no MSVC story; skip |

Three things the setuptools prototype hit that would have bitten a first
release, all fixed in `proto/pyproject.toml` with a comment at the line:

1. **The sdist silently omitted `nanoarrow.h`.** setuptools puts extension
   *sources* into the sdist, not headers, so `uv build` (which builds the
   wheel from the sdist) failed with `nanoarrow/nanoarrow.h: No such file`.
   Fix: `depends = ["src/d2shim/vendor/nanoarrow/nanoarrow.h"]`. The CI
   workflow's `sdist` job installs from the sdist for this reason.
2. **The wheel carried the C sources** (+335 KB) because
   `include-package-data` defaults to true and the sources live under the
   package. Fix: `include-package-data = false` and an explicit
   `package-data` list (`VERSION`, `SHA512SUMS`, `LICENSE.txt`, `NOTICE.txt`).
3. **A stale `build/` directory reuses old objects when only flags change.**
   Adding `-DNANOARROW_DLL=` produced no change until `build/` was deleted;
   distutils compares source timestamps only. `build_and_test.sh`,
   `run_cibuildwheel.sh` and the workflow all `rm -rf build` first.

And one hygiene item: as built by default the module exported **89 symbols
— 71 of them nanoarrow's own `Arrow*` API** — because `nanoarrow.h` marks its
functions `visibility("default")`. `extra-compile-args =
["-fvisibility=hidden"]` plus `define-macros = [..., ["NANOARROW_DLL", ""]]`
brings that to exactly one, `PyInit__nashim` (`nm -D` on the built `.so`).
Python loads extensions `RTLD_LOCAL`, so a second nanoarrow in the process
(the `nanoarrow` PyPI package, say) was never likely to collide, but a binary
that re-exports a third party's whole API is the first thing a reviewer
flags. MSVC ignores the GCC flag with a warning and exports nothing by
default, so the setting is safe to leave unconditional (untested on MSVC).

**Why a real extension module rather than a plain `.so` loaded by ctypes**
(the experiment's approach, and what three of the four alternatives do): the
shim has no Python in it, and turning it into a module costs 88 lines of C
that use four limited-API calls. In return, every tool in the chain — pip,
`auditwheel`, `delocate`, `abi3audit`, cibuildwheel, MSVC's `/EXPORT` — sees
a normal extension; a missing binary is a normal `ImportError` (§5); and no
platform needs a symbol-export mechanism because the sm_* addresses are
handed to Python as integers, which is exactly the form the numba intrinsic
already wants (a function pointer passed as a kernel *argument*, so the
kernel disk-caches). The price is `Python.h` at build time — irrelevant in
the manylinux container and on GitHub runners, relevant to
compile-on-install (§5).

---

## 3. The wheel matrix

The library links only libc (`readelf -d`: `libc.so.6`, `libpthread.so.0`),
touches no Python ABI beyond the four limited-API calls, and nanoarrow itself
publishes 59 PyPI wheels for 0.9.0 including `win32`/`win_amd64`, so the C is
known to compile under MSVC. Polars and numba — the only runtime
dependencies that matter — ship wheels for every row below.

| platform | wheel tag | runner | tested here | notes |
|---|---|---|---|---|
| Linux x86_64 glibc | `cp310-abi3-manylinux_2_28_x86_64` | `ubuntu-24.04` | **yes — the full cibuildwheel leg ran in podman** (`.tmp/cibuildwheel.log`): build 80 s, `auditwheel repair` → tagged `manylinux2014 / manylinux_2_17 / manylinux_2_28` (only `GLIBC_2.2`, `2.14` referenced, so it will install on RHEL 7+), `abi3audit --strict` → `is_abi3: true, baseline 3.10`, 44 tests pass inside the container; **3 min 08 s total**, 177 KB wheel | the bank's likely target |
| Linux aarch64 glibc | `cp310-abi3-manylinux_2_28_aarch64` | `ubuntu-24.04-arm` | **no** (x86 box; QEMU emulation not attempted) | native arm runners are free on public repos and avoid the ~10× QEMU penalty |
| macOS x86_64 | `cp310-abi3-macosx_11_0_x86_64` | `macos-13` | **no** | `MACOSX_DEPLOYMENT_TARGET=11.0` set to match polars/numba floors; `delocate` runs automatically |
| macOS arm64 | `cp310-abi3-macosx_11_0_arm64` | `macos-14` | **no** | developer laptops |
| Windows AMD64 | `cp310-abi3-win_amd64` | `windows-2022` | **no** — see the mingw partial evidence below | MSVC 2022; no `win32` (`skip = "*-win32"`), no ARM64 Windows |
| musllinux (Alpine) | — | — | — | deliberately skipped: falls back to pure numba (§5); add later if a musl image ever appears |
| free-threaded `cp3XYt` | — | — | — | cannot use abi3 wheels; numba does not support it; off by default in cibuildwheel 4 |

Five wheels plus one sdist. Because the tag is abi3, the matrix does not
grow with Python versions: a new CPython release needs no rebuild, only a
test run.

**Does the shim compile on Windows?** Read for MSVC idioms, `nashim.c` is
plain C99: `<stdint.h>` types, `size_t`, `int64_t`, no `ssize_t`, no
`__attribute__`, no VLAs, no statement expressions, declarations inside
`case` blocks are braced. `nanoarrow.h` carries explicit `_MSC_VER` branches
for its attribute macros (lines 223–233) and `_WIN32` handling for DLL
exports (1253–1263). The new `nashim_module.c` uses only `PyModule_Create`,
`PyModule_AddObjectRef`, `PyModule_AddIntConstant`,
`PyModule_AddStringConstant` and `PyLong_FromVoidPtr` — all in the 3.10
limited API, all in `python3.dll`. **What could be checked from Linux:**
`mingw_check.sh` cross-compiled `nashim.c` + `nanoarrow.c` for Windows x86_64
with mingw-w64 GCC 16 in a Fedora container — **0 warnings at
`-Wall -Wextra -pedantic -std=c99`**, a 296 KB DLL importing only
`KERNEL32.dll` and `msvcrt.dll`. That exercises the `_WIN32` code paths and
the Windows C runtime headers, **not MSVC's front end, not `python3.dll`
linking, and not the setuptools MSVC driver**. Nothing in the shim is expected
to change for MSVC; the untested claim is precisely "the `windows-2022`
cibuildwheel leg is green", and the first run of `ci/wheels.yml` is the test.

---

## 4. The fallback: pure numba vs compiled, measured

Both backends in one process, as shipped in the prototype wheel, the same
polars Series objects, a 6-node tree with three STR nodes (contains / prefix
/ exact) so every leaf is reachable, realistic merchant descriptors, 5 %
null. Compiled = `sm_get_string_checked` (bounds-checked, matching the pure
kernel's behaviour on corrupt input, `validate(DEFAULT)` once per call);
"unchecked" = nanoarrow's `GetStringUnsafe`. `bench_fallback.py`,
`.tmp/bench_fallback.log`; shared 28-core box, median / min of blocks.

**Identical answers first**: 1 M rows single-chunk, 1 M rows two-chunk,
500 k rows from a zero-copy slice with a non-zero Arrow offset — `np.array_equal`
true for all three, on top of the 44 unit tests (which also cover 0 rows,
all-null, 1-row, multi-chunk, sliced, and a 3000-row random column in all
four match modes).

| rows | pure numba (µs) | compiled, checked | compiled, unchecked | pure ÷ compiled |
|---|---|---|---|---|
| 1 | 43.9 / 42.4 | **26.3 / 25.8** | 24.4 / 24.0 | 1.67× |
| 100 | 56.7 / 56.2 | **32.4 / 32.3** | 30.9 / 30.7 | 1.75× |
| 1 000 | 116 / 116 | **101 / 101** | 99 / 99 | 1.15× |
| 1 000 000 | **73.1 ms / 72.7** | 78.6 / 76.8 ms | 78.5 / 77.2 ms | 0.93× |

The picture is the one the nanoarrow experiment already drew: the whole
difference is the Python-side import handshake (the pure path's ctypes loop
plus `np.ctypeslib` calls vs one C call), which is a fixed ~17 µs; per row
the compiled path pays a function call through a pointer and loses at scale.
On the 353 µs single-record path measured for the incumbent engine, the
fallback costs 5 %; against the 60 µs single-record target it is the
difference between 73 % and 44 % of budget — so the fallback is *acceptable*
but not *the same*, and the warning text says so.

**Is a two-implementation design worth it, or a trap?** Worth it, on three
grounds, with one condition.

* *It is not two implementations of the engine; it is two implementations
  of one 9–13 line function.* Both kernels import the tree walk
  (LEAF/CMP/IS_TRUE), the byte matcher `_match_at`, the pattern table and
  the chunk loop from `pure.py`; `kernel_compiled.py` is 97 lines of which
  the STR node is the only difference. What can drift is the string
  lookup and the treatment of corrupt input, and the equivalence suite is
  built to catch exactly that.
* *The fallback is the install story.* It is what runs on any platform
  without a wheel (§5), during the weeks between a new CPython and its
  numba release, on an air-gapped box, and in the CI `fallback` job. Without
  it, "no wheel" means "decider2 does not install", and the project has
  just made its first binary a hard dependency of a pure-Python engine.
* *It makes the nanoarrow decision reversible.* The previous experiment
  recommended against nanoarrow on cost grounds; if that judgement wins
  later, the pure path is already the shipped code and the binary is
  deleted, not rewritten.

The condition: **the two must never be allowed to differ in contract.**
Concretely — the compiled path defaults to the checked accessor (so both
produce `ERR_LEAF` on the same corruptions; the unchecked one segfaults
where pure returns an error leaf, per the experiment's §6), the suite runs
both backends against a Python reference in every CI job, and the `fallback`
job runs the whole test-suite with `DECIDER2_STRINGS=pure` so the pure path
is exercised even on platforms that have a wheel. Skip any of these and it
is a trap: the fallback rots unnoticed until the day it is needed.

---

## 5. Install-time behaviour on a platform with no wheel

pip falls back to the sdist, and what happens next depends on the target.
Every case below was run (`simulate_missing.sh`, `.tmp/simulate_missing.log`).

| case | what the user sees | exit |
|---|---|---|
| **A. sdist, no C compiler on PATH** (default config: extension required) | `error: [Errno 2] No such file or directory: 'cc'` from the build backend, then uv/pip's *"Build failures usually indicate a problem with the package or the build environment"* | **install fails** |
| **A2. sdist, compiler present, no Python headers** — this very dev box's `/usr/bin/python3.14` | `src/d2shim/c/nashim_module.c:19:10: fatal error: Python.h: No such file` | **install fails** |
| **B. sdist, no compiler, `optional = true` on the extension** | install succeeds; the package directory has no `_nashim*.so`; the first import prints one `FallbackWarning: d2shim: compiled string backend unavailable (ImportError: cannot import name '_nashim' ...); using the pure-numba fallback. Answers are identical; single-record calls cost ~2x more. Set D2SHIM_BACKEND=compiled to make this an error, or D2SHIM_BACKEND=pure to silence this warning.` | **works, pure** |
| **C. sdist, compiler and headers present** (compile on install) | 4.2 s: isolated env, fetch setuptools, compile three C files, install; `which() -> compiled` | **works, compiled** |
| **D. wheel installed, extension unloadable at runtime** (`sys.modules` trick; also a truncated `.so`: `ImportError: ... _nashim.abi3.so: file too short`) | same one-line `FallbackWarning` with the real cause embedded | **works, pure** |
| **D with `D2SHIM_BACKEND=compiled`** | `ImportError: D2SHIM_BACKEND=compiled but the compiled string backend is not available on Linux-...-glibc2.43 / CPython 3.14.5: <cause>. Either install a wheel built for this platform, or unset D2SHIM_BACKEND to accept the pure-numba fallback.` | **fails loudly** |

Is compile-on-install realistic in a bank? Case A2 answers it: it needs a C
compiler *and* the Python development headers, and a developer workstation
in this very project has the compiler but not the headers. Official
`python:3.x` Docker images have both; `python:3.x-slim` and UBI/RHEL images
have neither by default. It should be treated as a developer convenience,
never as the production path.

**The prototype's policy** (`strings.py`, 108 lines; prototyped as
`D2SHIM_BACKEND`, would be `DECIDER2_STRINGS`):

* unset → compiled if it imports, else pure with **one** `FallbackWarning`
  per process (a `RuntimeWarning` subclass, so `-W error::RuntimeWarning`
  turns it into a failure in test suites);
* `compiled` → compiled or `ImportError` with the platform, interpreter and
  underlying cause in the message — **set this in production images**;
* `pure` → never touch the extension — how the fallback is tested on
  platforms that have a wheel.

The decision is made once (`lru_cache`), and `diagnose()` returns what a
`decider2 doctor` subcommand would print: backend, reason, extension path,
nanoarrow version, the vendored pin, interpreter and platform. The
recommendation on `optional = true` (case B): **do not ship it.** It turns a
build failure into a warning that scrolls past in a Docker build log, which
is the silent-degradation risk from the front page in its purest form. Let
the sdist fail (A) and let platforms without a wheel install with
`--only-binary :all:` or get an explicit pure-only wheel if one is ever
wanted; the runtime fallback (D) covers the genuinely surprising cases.

---

## 6. Reproducibility and provenance: is the sha512 pin good enough?

What is there today: `vendor/VERSION` records the release (0.9.0,
2026-07-31), the tarball's sha512, the regeneration command
(`ci/scripts/bundle.py`) and the files kept. What was verified here
(`provenance_check.sh`, `.tmp/provenance.log`):

1. The tarball at `archive.apache.org/dist/arrow/apache-arrow-nanoarrow-0.9.0/`
   hashes to exactly the pinned sha512.
2. Upstream publishes a detached PGP signature (`.asc`) and a `.sha512`
   next to it (both HTTP 200) — the pin can be tied to the Apache release
   KEYS, not just to "whatever was downloaded".
3. Running upstream's `bundle.py` on the unpacked tarball regenerates
   `nanoarrow.c` and `nanoarrow.h` **byte-identical** to the vendored copies;
   `LICENSE.txt` and `NOTICE.txt` are byte-identical to the tarball's.
4. `nanoarrow.h` says `NANOARROW_VERSION "0.9.0"`, and the built module
   reports it at runtime (`diagnose()["nanoarrow"]`).

So the vendored copy is fully reproducible from a signed upstream release by
a documented, mechanical procedure. That is the strong part. What a bank's
reviewer would still ask for, and what the prototype adds:

| reviewer's question | answer / what was added |
|---|---|
| "The pin is of the tarball; what proves the files *in the tree* came from it?" | `vendor/SHA512SUMS` (new) hashes the four vendored files; `sha512sum -c` in the CI `provenance` job and in `provenance_check.sh` step 4. Regeneration (step 3) is the stronger check but needs network; the manifest is the offline one. |
| "Licence obligations?" | Apache-2.0 §4: keep the licence, keep the NOTICE, mark modified files (none are modified — the amalgamation is upstream's own output). PEP 639 `license-files` puts `LICENSE.txt` and `NOTICE.txt` into `<dist>.dist-info/licenses/` of every wheel and at the sdist root — verified in the wheel listing. decider2's own licence field should add `AND Apache-2.0` for the vendored portion; the project's `LICENSE` file is unaffected. |
| "How would a CVE in nanoarrow be noticed?" | **It would not, by any scanner the bank runs on Python packages.** Vendored C is invisible to `pip-audit`, Dependabot, Safety and SBOM tools that read `METADATA`: they see `decider2`, never `nanoarrow`. OSV today lists **zero** advisories for nanoarrow (PyPI or the GitHub repo; pyarrow has 9, for calibration), so nothing has been missed yet, but the process has to be explicit: (a) declare it in the SBOM — a CycloneDX component `pkg:github/apache/arrow-nanoarrow@0.9.0` with the sha512, generated from `VERSION` in the release job; (b) subscribe to `dev@arrow.apache.org` announcements / the GitHub security advisories for `apache/arrow-nanoarrow`; (c) the weekly CI cron is the place to add an OSV query against that purl. |
| "How is it patched?" | The five-step procedure in `VERSION` (download, verify, `bundle.py`, copy, rebuild) plus: update `SHA512SUMS`, bump the sha in `VERSION`, run `provenance_check.sh`, cut a release — the wheels rebuild from the sdist, there is no separate binary to patch. nanoarrow ships roughly two releases a year (0.6.0 Oct 2024, 0.7.0 Jul 2025, 0.8.0 Feb 2026, 0.9.0 Aug 2026 on PyPI); the shim uses 12 stable API functions and reads struct sizes at runtime, so an upgrade is expected to be a recompile. |
| "Who else vouches for the binary?" | The wheel's `.so` is produced in a container pinned by digest by cibuildwheel (`quay.io/pypa/manylinux_2_28_x86_64@sha256:5339...` in the log), by a public action, from the sdist; cibuildwheel prints the wheel's SHA256. Attestations (PEP 740 / GitHub artifact attestations) can be attached in the `collect` job when the project has a publishing target. The build is not bit-for-bit reproducible across toolchains and this note does not claim it. |
| "Is the vendored code auditable?" | 9,183 lines, exactly upstream's generated output, diffable against any nanoarrow version by re-running `bundle.py`; the project's own C is 6.7 KB + 88 lines, and only 37 of those lines (the `sm_get_string_checked` body, marked BEGIN/END in `nashim.c`) contain Arrow-layout knowledge. |

Verdict: the pin is the right *shape* — a signed upstream release, a
mechanical regeneration — and with the in-tree manifest, the licence files
in the wheel and an SBOM entry it is what a reviewer should accept. The
gap that no hash closes is the CVE feed, because vendoring takes the
library out of every scanner's field of view; that is a standing process
cost of the binary, not a one-off.

---

## 7. CI: what the matrix needs and what it costs

`ci/wheels.yml` (not installed in `.github/`). Jobs: `provenance` (hash
manifest + header version = pin), `sdist` (builds it, then installs a wheel
*from* it — the check that would have caught the missing-header bug),
`wheels` × 5 runners via cibuildwheel, `fallback` (full tests with the pure
backend), `collect` (exactly five `cp310-abi3` wheels + sdist, or fail).
Triggers: tags, PRs that touch `c/`, `vendor/` or `pyproject.toml`, a weekly
cron, manual. All wheel selection lives in `[tool.cibuildwheel]` in
`pyproject.toml` (11 lines), so the local `run_cibuildwheel.sh` and CI build
the same thing.

Cost, with the one measured leg as the anchor (Linux x86_64 in podman here:
3 min 08 s, of which image pull is excluded because it was cached, build
80 s, test-venv + polars/numba install + 44 tests ~100 s):

| job | runner | estimate |
|---|---|---|
| provenance + sdist + fallback | ubuntu | ~4 min combined |
| wheel Linux x86_64 | ubuntu-24.04 | ~4 min (add ~1 min image pull) |
| wheel Linux aarch64 | ubuntu-24.04-arm | ~4 min |
| wheel macOS x86_64 / arm64 | macos-13 / macos-14 | ~3–4 min each |
| wheel Windows | windows-2022 | ~6 min (MSVC and the polars/numba install are the slow parts) |
| **total** | | **~25 runner-minutes, ~6 min wall** (parallel) |

Billing: free on a public repo; on a private repo the multipliers (Linux 1×,
Windows 2×, macOS 10×) make it **~100 billed minutes per run**, which is why
the workflow runs on tags, relevant PRs and a weekly cron rather than every
push. Two cibuildwheel details learned the hard way, both commented in the
workflow: `test-sources` resolves against the *current directory*, not
`package-dir` (so run cibuildwheel from inside `decider2/`), and
`free-threaded-support` is no longer a config key in cibuildwheel 4.

---

## 8. What this box could not test — stated plainly

* **macOS (both arches) and Windows: not built, not tested.** The mingw
  cross-compile (§3) is partial evidence for Windows C portability only; MSVC,
  `python3.dll` linking, `delocate`, and the `MACOSX_DEPLOYMENT_TARGET` floor
  are unverified until `ci/wheels.yml` runs once.
* **Linux aarch64: not built.** No QEMU run was attempted.
* **A real RHEL 8/9 install of the manylinux wheel**: not done; the evidence
  is auditwheel's symbol analysis (`GLIBC_2.14` max) and the tags it granted.
* **The CI workflow file has not been executed** — it is written from the
  measured local leg and cibuildwheel's documented action interface.
* **Bit-for-bit reproducibility of the `.so`** across toolchains is not
  claimed; provenance is of the *source*.
* **Timings** are from a shared box with other agents running; the fallback
  benchmark was run twice with consistent results (±3 %), both runs in
  `.tmp/bench_fallback.log` history.
* **The prototype is `d2shim`, not decider2.** Moving it in means: the three
  `+ binary` blocks into `decider2/pyproject.toml` (with paths under
  `src/decider2/`), `nashim_module.c`'s module name, and renaming
  `D2SHIM_BACKEND` → `DECIDER2_STRINGS`. None of decider2's existing config
  changes.

---

## Decision taken (owner, this session)

**Ship the compiled nanoarrow shim as the ONLY implementation. Do not ship the
pure-numba fallback.**

This overrides §3 of this document, which recommended keeping the fallback
under a `DECIDER2_STRINGS` policy. The reasoning:

The measurements here show the binary is not buying speed — 17 µs per single
record (on a path where ~104 µs goes to per-call Python), and it *loses* 7% at
1M rows because its per-row call cannot inline. What the binary buys is the
project owning **zero lines of Arrow layout knowledge**, which was the whole
point of adopting nanoarrow.

Keeping the fallback would undo exactly that: the pure kernel is 42 lines of
hand-decoded Arrow layout, kept forever as a second implementation. The project
would pay the wheel matrix *and* keep the code it adopted nanoarrow to delete.

Consequences accepted:
- No install on a platform outside the 5-wheel matrix. The sdist needs a
  compiler; without one, install fails — loudly, by design.
- `optional=true` on the extension is NOT to be used; silent degradation to a
  slower path is worse than a failed install.
- The container build pipeline in `ci/wheels.yml` becomes load-bearing and must
  run before any release, since this dev box's GLIBC_2.38 build cannot be
  retagged `manylinux_2_28`.
- `proto/src/d2shim/strings.py`'s `auto`/`pure` backends and `pure.py` should be
  dropped from the Stage 1 implementation; `diagnose()` is still worth keeping
  for a `decider2 doctor` command.
