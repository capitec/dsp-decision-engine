"""Count the lines of Arrow-LAYOUT knowledge each approach makes the project
own, and print them so the count can be audited. A line counts if it encodes
a fact from the Arrow columnar spec: which buffer is which, the 16-byte view
shape, the inline/reference split at 12 bytes, the (buffer_index, offset)
pair, the validity bitmap's bit order, the offsets-buffer convention, how a
slice's `offset` applies. Plumbing (ctypes struct definitions of the C Data
Interface, numba intrinsics that load a byte, tree walking) does not count."""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.normpath(os.path.join(HERE, "..", "arrow-strings-in-tree"))


def lines(path):
    with open(path) as f:
        return f.read().splitlines()


def region(src, start_pat, end_pat, inclusive_end=True):
    """Lines from the first match of start_pat to the next match of end_pat."""
    s = next(i for i, ln in enumerate(src) if re.search(start_pat, ln))
    e = next(i for i in range(s + 1, len(src)) if re.search(end_pat, src[i]))
    return list(range(s, e + 1 if inclusive_end else e))


def code_only(src, idx):
    return [i for i in idx if src[i].strip() and not src[i].strip().startswith("#")]


def show(label, path, idx, src):
    print(f"\n### {label}: {len(idx)} lines ({os.path.relpath(path, HERE)})")
    for i in idx:
        print(f"  {i+1:4}: {src[i]}")
    return len(idx)


def main():
    total = {}
    # ---- A: incumbent ----
    k = lines(os.path.join(A, "kernel.py"))
    a = 0
    a += show("A kernel.py: _view (decode a 16-byte binview)", os.path.join(A, "kernel.py"),
              code_only(k, region(k, r"^def _view\(", r"return ln, bi, off")), k)
    a += show("A kernel.py: _is_valid (validity bitmap bit order)", os.path.join(A, "kernel.py"),
              code_only(k, region(k, r"^def _is_valid\(", r"return \(byte >>")), k)
    a += show("A kernel.py: STR node inline/reference split + bounds", os.path.join(A, "kernel.py"),
              code_only(k, region(k, r"va = str_views_addr\[col\]", r"s_addr = str_data_addr\[slot\]")), k)
    a += show("A kernel.py: string_tables slice handling (views + 16*offset, validity bit offset)",
              os.path.join(A, "kernel.py"),
              code_only(k, region(k, r"views\[k\] = c.views \+ 16 \* c.offset", r"valid_off\[k\] = c.offset")), k)
    ac = lines(os.path.join(A, "arrowc.py"))
    a += show("A arrowc.py: buffer roles of a 'vu' array (validity, views, variadic data, sizes)",
              os.path.join(A, "arrowc.py"),
              code_only(ac, region(ac, r'if fmt == "vu":', r"data_sizes=\[int\(x\) for x in sizes\]")), ac)
    total["A (hand-decoded numba)"] = a
    # ---- B: nanoarrow per row ----
    c = lines(os.path.join(HERE, "c", "nashim.c"))
    b_unsafe = 0  # sm_get_string wraps nanoarrow; no layout facts
    show("B nashim.c: sm_get_string (wraps nanoarrow; layout lines = 0)", os.path.join(HERE, "c", "nashim.c"),
         code_only(c, region(c, r"^int64_t sm_get_string\(", r"^}")), c)
    bchk = show("B-checked nashim.c: the hand-written bounds check (BEGIN..END)", os.path.join(HERE, "c", "nashim.c"),
                code_only(c, region(c, r"BEGIN layout knowledge", r"END layout knowledge")), c)
    total["B (nanoarrow per row, unsafe accessor)"] = b_unsafe
    total["B-checked (nanoarrow + hand-written bounds check in C)"] = bchk
    # ---- C ----
    total["C (nanoarrow validate + A's numba decode)"] = a
    print("\n### TOTALS (lines of Arrow-layout knowledge the project owns)")
    for name, n in total.items():
        print(f"  {name:<58} {n:4}")
    print(f"  vendored nanoarrow (a dependency, not owned): {len(lines(os.path.join(HERE,'vendor','nanoarrow.c')))} + "
          f"{len(lines(os.path.join(HERE,'vendor','nanoarrow','nanoarrow.h')))} lines")
    import json
    with open(os.path.join(HERE, "results.jsonl"), "a") as f:
        f.write(json.dumps({"probe": "layout_lines", **total}) + "\n")


if __name__ == "__main__":
    main()
