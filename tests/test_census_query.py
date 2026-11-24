import unittest

import numpy as np
from experiments.reuse import census_query as cq

NET_DEV = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 5000      10    0    0    0     0          0         0     5000      10    0    0    0     0       0          0
  eth0: 120000    90    0    0    0     0          0         0     3000      20    0    0    0     0       0          0
  eth1:   7000     5    0    0    0     0          0         0      100       1    0    0    0     0       0          0
"""


class ReceivedBytes(unittest.TestCase):
    def test_sums_every_interface_but_loopback(self):
        self.assertEqual(cq.received_bytes(NET_DEV), 127000)

    def test_only_loopback_is_an_error(self):
        lines = NET_DEV.splitlines()
        with self.assertRaisesRegex(RuntimeError, "besides loopback"):
            cq.received_bytes("\n".join(lines[:3]))


class GeneValues(unittest.TestCase):
    def test_cells_without_an_entry_count_zero_and_values_are_log_per_10k(self):
        joinids = np.array([10, 20, 30])
        totals = np.array([1000.0, 2000.0, 4000.0])
        values = cq.gene_values(joinids, totals, np.array([30, 10]), np.array([4.0, 1.0]))
        np.testing.assert_allclose(values, [np.log1p(10.0), 0.0, np.log1p(10.0)])

    def test_counts_for_cells_not_asked_for_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "not asked for"):
            cq.gene_values(np.array([10, 20]), np.ones(2), np.array([15]), np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "not asked for"):
            cq.gene_values(np.array([10, 20]), np.ones(2), np.array([25]), np.array([1.0]))

    def test_unsorted_cells_are_an_error(self):
        with self.assertRaisesRegex(ValueError, "sorted"):
            cq.gene_values(np.array([20, 10]), np.ones(2), np.array([10]), np.array([1.0]))


class SparseEntries(unittest.TestCase):
    def test_an_arrow_table_of_several_chunks_becomes_cells_and_values(self):
        import pyarrow as pa

        chunk = pa.table({"soma_dim_0": [10, 30], "soma_dim_1": [0, 0], "soma_data": [1, 4]})
        later = pa.table({"soma_dim_0": [40], "soma_dim_1": [0], "soma_data": [2]})
        entries, values = cq.sparse_entries(pa.concat_tables([chunk, later]))
        self.assertEqual(entries.tolist(), [10, 30, 40])
        self.assertEqual(values.dtype, np.float64)
        self.assertEqual(values.tolist(), [1.0, 4.0, 2.0])


if __name__ == "__main__":
    unittest.main()
