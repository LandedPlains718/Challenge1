#!/usr/bin/env python3
"""Normalize stringified Python dicts in address / primary_naics / secondary_naics."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

NESTED_FIELDS = ("address", "primary_naics", "secondary_naics")


def parse_maybe(value):
    if value is None or isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or text.lower() == "null":
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value

    if isinstance(parsed, (dict, list)):
        return parsed
    return value


def fix_record(record: dict) -> tuple[dict, list[str]]:
    fixed_fields: list[str] = []
    for field in NESTED_FIELDS:
        if field not in record:
            continue
        original = record[field]
        parsed = parse_maybe(original)
        if parsed is not original:
            record[field] = parsed
            fixed_fields.append(field)
    return record, fixed_fields


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_src = root / "data" / "companies.jsonl"
    src = Path(sys.argv[1] if len(sys.argv) > 1 else default_src)
    dst = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else src.with_name(f"{src.stem}_fixed.jsonl")
    )

    fixed_rows = 0
    field_counts = {f: 0 for f in NESTED_FIELDS}

    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip line {line_no}: {exc}", file=sys.stderr)
                continue

            record, fixed_fields = fix_record(record)
            if fixed_fields:
                fixed_rows += 1
                for field in fixed_fields:
                    field_counts[field] += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {dst}")
    print(f"rows with at least one fix: {fixed_rows}")
    for field, count in field_counts.items():
        print(f"  {field}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
