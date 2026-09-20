"""Validate extraction output chunks against their source chunks.

Usage: .venv/bin/python src/validate_chunks.py <src_dir> <out_dir>
Prints per-chunk status; exits nonzero if any chunk is missing/invalid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def validate(src_dir: str, out_dir: str) -> int:
    bad = 0
    done = 0
    pending = []
    for src in sorted(Path(src_dir).glob("chunk_*.json")):
        out = Path(out_dir) / src.name
        if not out.exists():
            pending.append(src.name)
            continue
        src_uids = {r["uid"] for r in json.load(open(src))}
        try:
            rows = json.load(open(out))
        except json.JSONDecodeError as e:
            print(f"INVALID JSON {out.name}: {e}")
            bad += 1
            continue
        out_uids = {r.get("uid") for r in rows}
        problems: list[str] = []
        if missing := src_uids - out_uids:
            problems.append(f"missing {len(missing)} uids: {sorted(missing)[:2]}...")
        if extra := out_uids - src_uids:
            problems.append(f"extra uids: {sorted(extra)[:2]}")
        for r in rows:
            keys = set(r) - {"uid"}
            if keys != set(TARGETS):
                problems.append(f"bad keys for {r.get('uid', '?')[-8:]}")
                break
            if not all(
                isinstance(r[t], (int, float)) and 0.0 <= r[t] <= 1.0 for t in TARGETS
            ):
                problems.append(f"out-of-range value for {r.get('uid', '?')[-8:]}")
                break
        if problems:
            print(f"BAD {out.name}: {'; '.join(problems)}")
            bad += 1
        else:
            done += 1
    print(f"\nvalid: {done}  bad: {bad}  pending: {len(pending)}")
    if pending:
        print("next pending:", ", ".join(pending[:12]))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(validate(sys.argv[1], sys.argv[2]))
