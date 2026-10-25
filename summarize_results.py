#!/usr/bin/env python3
"""Summarize tracking.log files for seed_0, seed_1, and seed_2.

Run from the repository root:
    python summarize_results.py

Expected layout:
    results/<dataset>/<data_type>/<algorithm>/<exp_name>/seed_<seed>/tracking.log
"""
import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "summary"
VALID_SEEDS = {0, 1, 2}
METRICS = [
    "acc-flipping", "acc-normal",
    "f1-flipping", "f1-normal",
    "recall-flipping", "recall-normal",
    "precision-flipping", "precision-normal",
]


def search_value(pattern, text, default=None):
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else default


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    parts = log_path.parts

    # Parse standard directory layout.
    dataset = data_type = algorithm = exp_name = ""
    seed = None
    try:
        result_idx = parts.index("results")
        tail = list(parts[result_idx + 1:-1])  # remove tracking.log
        if len(tail) >= 4:
            dataset, data_type, algorithm, exp_name = tail[:4]
        if tail and re.fullmatch(r"seed_\d+", tail[-1]):
            seed = int(tail[-1].split("_", 1)[1])
    except ValueError:
        pass

    # Metadata in the log can override/fill path metadata.
    dataset = search_value(r"^\s*Dataset\s*:\s*(.+?)\s*$", text, dataset) or dataset
    algorithm = search_value(r"^\s*Algorithm\s*:\s*(.+?)\s*$", text, algorithm) or algorithm
    if seed is None:
        seed_text = search_value(r"^\s*Seed\s*:\s*(\d+)\s*$", text)
        if seed_text is not None:
            seed = int(seed_text)

    noise_type = search_value(r"Noise\s*type\s*:\s*([^|\r\n]+)", text, "") or ""
    noise_ratio = search_value(r"\bratio\s*:\s*([0-9.eE+-]+)", text, "") or ""
    noise_parameter = search_value(
        r"\b(?:noise\s*parameter|parameter|epsilon|rc|rs)\s*:\s*([0-9.eE+-]+)",
        text, "",
    ) or ""

    # Extract the last "Best performance metrics" block.
    matches = list(re.finditer(r"Best\s+performance\s+metrics", text, re.I))
    if not matches:
        print(f"[SKIP] No Best performance metrics block: {log_path}")
        return None

    start = matches[-1].end()
    remaining = text[start:]
    endings = [
        m.start() for pat in (
            r"\n\s*-{5,}\s*Round\s+number\s*:",
            r"\n\s*Round\s+number\s*:",
            r"\n\s*Best\s+performance\s+metrics",
        )
        for m in re.finditer(pat, remaining, re.I)
    ]
    block = remaining[:min(endings)] if endings else remaining

    values = {}
    for metric in METRICS:
        pattern = (
            r"^\s*" + re.escape(metric)
            + r"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        )
        value = search_value(pattern, block)
        values[metric] = float(value) if value is not None else None

    if all(v is None for v in values.values()):
        print(f"[SKIP] Metric labels not recognized in {log_path}")
        return None

    return {
        "dataset": dataset,
        "data_type": data_type,
