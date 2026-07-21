#!/usr/bin/env python
"""Manage quant learning queue status.

The queue is intentionally simple CSV so future agents can inspect and repair it.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path


EXTRA_FIELDS = [
    "learned_at",
    "review_status",
    "correction_status",
    "artifact",
]


def normalize_path(value: str) -> str:
    return value.replace("/", "\\").strip().lower()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
        for row in rows:
            row.setdefault(field, "")
    return rows, fields


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> str:
    by_status = Counter(row.get("status", "") for row in rows)
    by_priority = Counter(row.get("priority_hint", "") for row in rows)
    by_review = Counter(row.get("review_status", "") for row in rows)
    lines = ["status:"]
    lines.extend(f"  {key or 'blank'}={value}" for key, value in sorted(by_status.items()))
    lines.append("priority:")
    lines.extend(f"  {key or 'blank'}={value}" for key, value in sorted(by_priority.items()))
    lines.append("review:")
    lines.extend(f"  {key or 'blank'}={value}" for key, value in sorted(by_review.items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Corpus root.")
    parser.add_argument("--queue", default="reading_queue.csv", help="Queue CSV path under root.")
    parser.add_argument("--mark", nargs="*", default=[], help="Relative paths to update.")
    parser.add_argument("--status", help="Set status for marked paths.")
    parser.add_argument("--review-status", help="Set review_status for marked paths.")
    parser.add_argument("--correction-status", help="Set correction_status for marked paths.")
    parser.add_argument("--artifact", help="Set artifact for marked paths.")
    parser.add_argument("--summary", action="store_true", help="Print queue summary.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    queue_path = root / args.queue
    rows, fields = read_rows(queue_path)

    if args.mark:
        targets = {normalize_path(path) for path in args.mark}
        now = datetime.now().isoformat(timespec="seconds")
        updated = 0
        for row in rows:
            if normalize_path(row.get("path", "")) not in targets:
                continue
            if args.status:
                row["status"] = args.status
                if args.status in {"learned", "reviewed", "corrected", "promoted"}:
                    row["learned_at"] = row.get("learned_at") or now
            if args.review_status:
                row["review_status"] = args.review_status
            if args.correction_status:
                row["correction_status"] = args.correction_status
            if args.artifact:
                row["artifact"] = args.artifact
            updated += 1
        write_rows(queue_path, rows, fields)
        print(f"updated={updated}")

    if args.summary:
        print(summarize(rows))


if __name__ == "__main__":
    main()
