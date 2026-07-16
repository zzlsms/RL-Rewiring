#!/usr/bin/env python3
"""Merge distributed AIME shards and report Pass@1 and Pass@k."""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute the standard unbiased Pass@k estimate."""
    if n < 0 or c < 0 or c > n or k < 1:
        raise ValueError(f"Invalid Pass@k inputs: n={n}, c={c}, k={k}")
    if n < k or c == 0:
        return 0.0
    if c == n or n - c < k:
        return 1.0

    failure_probability = 1.0
    for i in range(k):
        failure_probability *= (n - c - i) / (n - i)
    return 1.0 - failure_probability


def load_shards(result_dir: str) -> tuple[list[dict[str, Any]], list[str]]:
    files = sorted(glob.glob(os.path.join(result_dir, "shard_*.json")))
    if not files:
        raise FileNotFoundError(f"No shard_*.json files found in {result_dir}")

    payloads: list[dict[str, Any]] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to parse shard file: {path}") from exc
    return payloads, files


def validate_shards(payloads: list[dict[str, Any]]) -> None:
    configs = [payload.get("config") for payload in payloads]
    if all(config is None for config in configs):
        return  # Backward compatibility with result files released without config metadata.
    if any(config is None for config in configs):
        raise ValueError("Some shard files contain config metadata while others do not.")

    typed_configs = [config for config in configs if config is not None]
    consistency_keys = (
        "dataset",
        "split",
        "expected_problems",
        "num_shards",
        "samples_per_problem",
        "temperature",
        "top_p",
        "max_tokens",
    )
    for key in consistency_keys:
        values = {config.get(key) for config in typed_configs}
        if len(values) != 1:
            raise ValueError(f"Inconsistent shard configuration for '{key}': {values}")

    expected_shards = int(typed_configs[0]["num_shards"])
    shard_ids = [int(config["shard_id"]) for config in typed_configs]
    if sorted(shard_ids) != list(range(expected_shards)):
        raise ValueError(
            f"Incomplete or duplicated shard set: found {sorted(shard_ids)}, "
            f"expected {list(range(expected_shards))}"
        )

    failed = sum(int(payload.get("metrics", {}).get("failed_problems", 0)) for payload in payloads)
    if failed:
        raise ValueError(f"Shard files report {failed} failed problem(s).")


def aggregate_problems(payloads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    problem_map: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for item in payload.get("detailed_results", []):
            problem_text = str(item["problem"])
            stats = problem_map.setdefault(
                problem_text,
                {"text": problem_text, "correct": 0, "total": 0, "token_sum": 0.0},
            )

            correct = int(item.get("correct_count", 0))
            total = int(item.get("total_samples", 0))
            avg_tokens = float(item.get("avg_tokens", 0.0))
            if total < 0 or correct < 0 or correct > total:
                raise ValueError(
                    f"Invalid counts for problem {problem_text[:60]!r}: "
                    f"correct={correct}, total={total}"
                )

            stats["correct"] += correct
            stats["total"] += total
            stats["token_sum"] += avg_tokens * total

    if not problem_map:
        raise ValueError("Shard files contain no completed problems.")
    return problem_map


def build_report(problem_map: dict[str, dict[str, Any]], target_k: int) -> dict[str, Any]:
    problem_reports: list[dict[str, Any]] = []
    total_samples = 0
    total_tokens = 0.0

    for stats in problem_map.values():
        n = int(stats["total"])
        c = int(stats["correct"])
        pass_at_1 = c / n if n else 0.0
        pass_at_k = estimate_pass_at_k(n, c, target_k)
        problem_reports.append(
            {
                "problem": stats["text"],
                "total_samples": n,
                "correct_count": c,
                "pass@1": pass_at_1,
                f"pass@{target_k}": pass_at_k,
            }
        )
        total_samples += n
        total_tokens += float(stats["token_sum"])

    return {
        "num_problems": len(problem_reports),
        "total_sequences": total_samples,
        "average_token_length": total_tokens / total_samples if total_samples else 0.0,
        "pass@1": float(np.mean([item["pass@1"] for item in problem_reports])),
        f"pass@{target_k}": float(
            np.mean([item[f"pass@{target_k}"] for item in problem_reports])
        ),
        "problems": problem_reports,
    }


def print_report(report: dict[str, Any], target_k: int) -> None:
    width = 110
    print("\n" + "=" * width)
    print(f"{'Problem (first 60 chars)':<65} | {'N':>4} | {'Correct':>7} | {'Pass@1':>8} | Pass@{target_k}")
    print("-" * width)
    for item in report["problems"]:
        snippet = item["problem"].replace("\n", " ")[:60]
        print(
            f"{snippet:<65} | {item['total_samples']:>4} | "
            f"{item['correct_count']:>7} | {item['pass@1']:>8.2%} | "
            f"{item[f'pass@{target_k}']:>8.2%}"
        )

    print("=" * width)
    print("FINAL AGGREGATED REPORT")
    print(f"Total Unique Problems: {report['num_problems']}")
    print(f"Total Sequences:        {report['total_sequences']}")
    print(f"Average Token Length:   {report['average_token_length']:.2f}")
    print("-" * 40)
    print(f"Global Pass@1:          {report['pass@1']:.2%}")
    print(f"Global Pass@{target_k}:         {report[f'pass@{target_k}']:.2%}")
    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_dir",
        "--result-dir",
        type=str,
        default="outputs/aime24",
        help="Directory containing shard_*.json files.",
    )
    parser.add_argument(
        "--target_k",
        "--target-k",
        type=int,
        default=32,
        help="Target k for Pass@k calculation.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    payloads, files = load_shards(args.result_dir)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(files)} shard files.")
    validate_shards(payloads)
    report = build_report(aggregate_problems(payloads), args.target_k)
    print_report(report, args.target_k)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
