import unittest
import tests.test_scan as tscan
import tests.test_boruvka as tboruvka

scan_suite = unittest.TestLoader().loadTestsFromModule(tscan)
boruvka_suite = unittest.TestLoader().loadTestsFromModule(tboruvka)

# unittest.TextTestRunner(verbosity=2).run(scan_suite)
unittest.TextTestRunner(verbosity=2).run(boruvka_suite)
