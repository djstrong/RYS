from __future__ import annotations

import unittest

from src.utils.surrogate_utils import (
    count_vector_to_layers,
    counts_from_csv,
    counts_to_csv,
    key_to_count_vector,
    relative_overhead_from_counts,
    stable_quantile_bins,
)


class SurrogateUtilsTests(unittest.TestCase):
    def test_key_to_count_vector_roundtrip(self):
        key = (0, 1, 2, 1, 2, 3)
        counts = key_to_count_vector(key, num_layers=4)
        self.assertEqual(counts, [1, 2, 2, 1])
        decoded = count_vector_to_layers(counts, num_layers=4)
        self.assertEqual(decoded, [0, 1, 1, 2, 2, 3])

    def test_counts_csv_parse(self):
        counts = [1, 2, 3, 1]
        raw = counts_to_csv(counts)
        self.assertEqual(raw, "1,2,3,1")
        self.assertEqual(counts_from_csv(raw, expected_len=4), counts)

    def test_relative_overhead(self):
        counts = [1, 2, 2, 1]
        overhead = relative_overhead_from_counts(counts, num_layers=4)
        self.assertAlmostEqual(overhead, 0.5)

    def test_stable_quantile_bins(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        bins = stable_quantile_bins(values, bins=3)
        self.assertEqual(len(bins), 5)
        self.assertTrue(all(0 <= b < 3 for b in bins))


if __name__ == "__main__":
    unittest.main()

