"""JSONL helpers and matching of scored rows back to their pool positions."""

import json

ALPACA_PROMPT = ("Below is an instruction that describes a task. "
                 "Write a response that appropriately completes the request.\n\n"
                 "### Instruction:\n{instruction}\n\n### Response:")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_lines(path):
    """Raw lines of a JSONL file (no trailing empty line), so a selection can be written back byte for byte."""
    lines = open(path, encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def write_lines(path, lines):
    """Writes lines + newline; never overwrites an existing file."""
    with open(path, "x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def response_of(row):
    return row["response"] if "response" in row else row.get("output", "")


def prompt_of(row, template=None):
    """The prompt the scorer and the trainer see: the Alpaca template, or a custom template with {instruction} / {input}."""
    return (template or ALPACA_PROMPT).format_map({"instruction": row.get("instruction", ""), "input": row.get("input", "") or ""})


def row_key(row):
    return (str(row.get("instruction", "")).strip(), str(row.get("input") or "").strip(), str(response_of(row)).strip())


def match_to_pool(rows, pool):
    """Pool indices of `rows`, matched by text as a multiset (a pool may hold exact duplicates). Returns (indices, n_missing)."""
    left = {}
    for j, r in enumerate(pool):
        left.setdefault(row_key(r), []).append(j)
    out, missing = [], 0
    for r in rows:
        q = left.get(row_key(r))
        if q:
            out.append(q.pop(0))
        else:
            missing += 1
    return out, missing
