"""Validate extraction output chunks against their source chunks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from knee.common import TARGETS


def read_json(path: Path) -> list[dict]:
    """Reads one chunk file.

    Args:
        path: JSON file holding an array of records.

    Returns:
        The decoded records.

    Raises:
        json.JSONDecodeError: The file is not valid JSON.
    """
    with open(path) as f:
        return json.load(f)


def chunk_problems(src_uids: set[str], rows: list[dict]) -> list[str]:
    """Lists what is wrong with one output chunk.

    Args:
        src_uids: The study UIDs the source chunk asked for.
        rows: The records the extraction produced.

    Returns:
        One short message per problem, empty when the chunk is good.
    """
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
    return problems


def validate(src_dir: str, out_dir: str) -> int:
    """Checks every output chunk against its source chunk and prints a summary.

    Args:
        src_dir: Directory of chunk_*.json inputs, each an array of {"uid", ...}.
        out_dir: Directory of extraction outputs with the same file names.

    Returns:
        1 if any chunk is present but invalid, 0 otherwise. A chunk that has not been
        produced yet counts as pending, not bad.
    """
    bad = 0
    done = 0
    pending = []
    for src in sorted(Path(src_dir).glob("chunk_*.json")):
        out = Path(out_dir) / src.name
        if not out.exists():
            pending.append(src.name)
            continue
        src_uids = {r["uid"] for r in read_json(src)}
        try:
            rows = read_json(out)
        except json.JSONDecodeError as e:
            print(f"INVALID JSON {out.name}: {e}")
            bad += 1
            continue
        problems = chunk_problems(src_uids, rows)
        if problems:
            print(f"BAD {out.name}: {'; '.join(problems)}")
            bad += 1
        else:
            done += 1
    print(f"\nvalid: {done}  bad: {bad}  pending: {len(pending)}")
    if pending:
        print("next pending:", ", ".join(pending[:12]))
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    """Runs the validation from the command line.

    Args:
        argv: [src_dir, out_dir].

    Returns:
        The process exit status; 2 when the arguments are wrong.
    """
    if len(argv) != 2:
        print("usage: python -m knee.labels.validate_chunks <src_dir> <out_dir>")
        return 2
    return validate(argv[0], argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
