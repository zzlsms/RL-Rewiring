from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation_aime24_distributed import is_equiv, strip_math_answer
from scripts.summarize_aime24 import (
    aggregate_problems,
    build_report,
    estimate_pass_at_k,
    load_shards,
    validate_shards,
)


class EvaluationUtilityTests(unittest.TestCase):
    def test_aime_leading_zeros_are_equivalent(self) -> None:
        self.assertEqual(strip_math_answer("025"), "25")
        self.assertTrue(is_equiv("025", "25"))

    def test_pass_at_k_estimator(self) -> None:
        self.assertEqual(estimate_pass_at_k(4, 0, 4), 0.0)
        self.assertEqual(estimate_pass_at_k(4, 1, 4), 1.0)
        self.assertAlmostEqual(estimate_pass_at_k(4, 1, 1), 0.25)
        for n in range(1, 16):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    expected = (
                        1.0 - math.comb(n - c, k) / math.comb(n, k)
                        if n - c >= k
                        else 1.0
                    )
                    self.assertAlmostEqual(estimate_pass_at_k(n, c, k), expected)

    def test_complete_shard_set_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shard_id in range(2):
                payload = {
                    "config": {
                        "dataset": "demo",
                        "split": "train",
                        "expected_problems": 2,
                        "shard_id": shard_id,
                        "num_shards": 2,
                        "samples_per_problem": 4,
                    },
                    "metrics": {"failed_problems": 0},
                    "detailed_results": [
                        {
                            "problem": f"problem-{shard_id}",
                            "correct_count": shard_id,
                            "total_samples": 4,
                        }
                    ],
                }
                path = root / f"shard_{shard_id}_of_2_k4.json"
                path.write_text(json.dumps(payload), encoding="utf-8")

            payloads, paths = load_shards(str(root))
            validate_shards(payloads)
            report = build_report(aggregate_problems(payloads), target_k=4)
            self.assertEqual(len(paths), 2)
            self.assertEqual(report["num_problems"], 2)
            self.assertEqual(report["total_sequences"], 8)
            self.assertAlmostEqual(report["pass@1"], 0.125)


if __name__ == "__main__":
    unittest.main()
