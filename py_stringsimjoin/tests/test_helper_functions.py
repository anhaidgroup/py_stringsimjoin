import multiprocessing
import unittest

from nose.tools import assert_equal, assert_list_equal, raises
import pandas as pd

from py_stringsimjoin.utils.generic_helper import get_num_processes_to_launch


class GetNumProcessesToLaunchTestCases(unittest.TestCase):
    def setUp(self):
        self.cpu_count = multiprocessing.cpu_count()

    def test_n_jobs_minus_1(self):
        assert_equal(get_num_processes_to_launch(-1), self.cpu_count)

    def test_n_jobs_1(self):
        assert_equal(get_num_processes_to_launch(1), 1)

    def test_n_jobs_valid_negative_value(self):
        valid_neg_n_jobs = -1
        if self.cpu_count > 1:
            valid_neg_n_jobs -= 1 
        assert_equal(get_num_processes_to_launch(valid_neg_n_jobs), 
                     (self.cpu_count - 1) if self.cpu_count > 1 else self.cpu_count)

    def test_n_jobs_invalid_negative_value(self):
        invalid_neg_n_jobs = self.cpu_count * 2 * -1
        assert_equal(get_num_processes_to_launch(invalid_neg_n_jobs), 1)

    def test_n_jobs_valid_positive_value(self):
        assert_equal(get_num_processes_to_launch(2), 2)

    def test_n_jobs_above_cpu_count(self):
        assert_equal(get_num_processes_to_launch(self.cpu_count + 1),
                     self.cpu_count + 1)

    def test_n_jobs_0(self):
        assert_equal(get_num_processes_to_launch(0), 1)
