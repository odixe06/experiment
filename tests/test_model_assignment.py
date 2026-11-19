"""Assignment integrity, paired comparisons and GPU-shard invariance; standard library only."""
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from core.model_assignment import CASES, assignment, load_assignment, write_bundle


class AssignmentTest(unittest.TestCase):
    def test_exact_counts_coverage_and_training_rng_independence(self):
        expected = [(1603, 1603, 2137), (1603, 1603, 2137), (1069, 1069, 1069, 1068, 1068)]
        for case, values in zip(CASES, expected):
            original = assignment(case)
            self.assertEqual(set(original), set(range(5343)))
            self.assertEqual(Counter(original.values()), dict(zip(dict(CASES[case]), values)))
            for seed in (41, 42, 43):
                random.seed(seed)
                random.random()
                state = random.getstate()
                self.assertEqual(original, assignment(case))
                self.assertEqual(state, random.getstate())

    def test_shards_preserve_global_id_assignment(self):
        with tempfile.TemporaryDirectory() as temp:
            write_bundle(temp)
            for case in CASES:
                whole = load_assignment(temp, case)
                ids = list(whole)
                random.Random(98).shuffle(ids)  # stand-in for arbitrary scheduling by train size
                left = load_assignment(temp, case, ids[::2])
                right = load_assignment(temp, case, ids[1::2])
                self.assertFalse(set(left) & set(right))
                self.assertEqual(whole, left | right)
            for ids in ([0, 0], [-1], [5343]):
                with self.assertRaises(ValueError):
                    load_assignment(temp, "case1", ids)

    def test_tamper_rejected_and_existing_files_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            write_bundle(temp)
            path = Path(temp) / "case1.csv"
            original = path.read_bytes()
            write_bundle(temp)
            self.assertEqual(original, path.read_bytes())
            lines = original.decode().splitlines()
            # Swap two model labels, preserving both ID coverage and model counts.
            a = lines[1].split(",")
            other = next(i for i in range(2, len(lines)) if lines[i].split(",")[1] != a[1])
            b = lines[other].split(",")
            lines[1], lines[other] = f"{a[0]},{b[1]}", f"{b[0]},{a[1]}"
            corrupted = ("\n".join(lines) + "\n").encode()
            path.write_bytes(corrupted)
            with self.assertRaises(ValueError):
                load_assignment(temp, "case1")
            with self.assertRaises(ValueError):
                write_bundle(temp)
            self.assertEqual(corrupted, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
