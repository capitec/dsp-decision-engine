#!/usr/bin/env python3
"""Why does the fused kernel lose when the body is expensive?

Doc 01 4b says the fused body "loses LLVM auto-vectorisation (vector IR values
drop to 0 at M>=4)".  The sweep shows vector ops never reach 0, so this digs into
the actual hot loop: unroll factor, distinct vector registers live, and spills.
"""
import re, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from fusion_body_cost import build_fused, build_split_kernel

FMA = re.compile(r"^\s*(vfmadd\d*pd|vfmadd\d*sd|vmulpd|vmulsd|vaddpd|vaddsd)\b")
YMM = re.compile(r"\b(ymm\d+|zmm\d+)\b")
SPILL = re.compile(r"^\s*vmov[au]?p?[ds]{1,2}\s+[^,]*,\s*-?\d*\(%?rsp")
LABEL = re.compile(r"^\.?[A-Za-z_$.][\w$.]*:")
JMPBACK = re.compile(r"^\s*(jmp|jne|jae|jb|jl|jg|je|jbe|ja|jle|jge)\b\s+(\S+)")


def loops(asm):
    """Split asm into blocks and return blocks that are targets of a backward jump."""
    lines = asm.splitlines()
    pos = {}
    for i, ln in enumerate(lines):
        m = LABEL.match(ln.strip())
        if m:
            pos[ln.strip().rstrip(":")] = i
    out = []
    for i, ln in enumerate(lines):
        m = JMPBACK.match(ln)
        if m:
            tgt = m.group(2).lstrip(".").lstrip("%")
            for lbl, j in pos.items():
                if lbl.lstrip(".") == tgt and j < i:
                    out.append((lbl, j, i))
    return lines, out


def report(name, disp):
    asm = "\n".join(disp.inspect_asm().values())
    lines, ls = loops(asm)
    if not ls:
        print(f"{name}: no backward jump found ({len(lines)} asm lines)")
        return
    print(f"{name}:  ({len(lines)} asm lines, {len(ls)} loops)")
    seen = set()
    rows = []
    for lbl, a, b in sorted(ls, key=lambda t: t[1] - t[2]):
        if lbl in seen:
            continue
        seen.add(lbl)
        body = lines[a:b + 1]
        if len(body) < 12:
            continue
        fma = sum(1 for ln in body if FMA.match(ln))
        fma_p = sum(1 for ln in body if FMA.match(ln)
                    and FMA.match(ln).group(1).endswith("pd"))
        regs = set()
        for ln in body:
            regs.update(YMM.findall(ln))
        stack_refs = sum(1 for ln in body if re.search(r"-?\d+\(%rsp\)", ln))
        rows.append((lbl, len(body), fma_p, fma - fma_p, len(regs), stack_refs))
    print("     loop         lines  packedFP  scalarFP  ymm/zmm  %rsp refs")
    for lbl, n, fp, fs, nr, sp in rows[:4]:
        print(f"     {lbl:<12s} {n:5d}  {fp:8d}  {fs:8d}  {nr:7d}  {sp:9d}")


if __name__ == "__main__":
    one = np.zeros(1) + 0.5
    for body, br in (("heavy", "straight"), ("medium", "straight"),
                     ("trivial", "straight"), ("heavy", "branchy")):
        print(f"\n===== body={body} branch={br} =====")
        k, _ = build_split_kernel(body, br, 0)
        k(one, one, np.zeros(1), np.zeros(1))
        report("  SPLIT  (1 module/kernel)", k)
        for M in (4, 20):
            f, _ = build_fused(body, br, M)
            f(one, one, np.zeros(1))
            report(f"  FUSED  M={M}", f)
