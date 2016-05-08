import unittest

import pandas as pd

from py_stringsimjoin.filter.position_filter import PositionFilter
from py_stringsimjoin.utils.tokenizers import create_delimiter_tokenizer


# test PositionFilter.filter_pair method
class FilterPairTestCases(unittest.TestCase):
    def setUp(self):
        self.tokenizer = create_delimiter_tokenizer(' ')
        self.position_filter = PositionFilter(self.tokenizer,
                                              sim_measure_type='JACCARD',
                                              threshold=0.8)

    def test_to_be_pruned_pair(self):
        self.assertTrue(self.position_filter.filter_pair('aa bb cc dd ee',
                                                         'cc dd'))

    def test_to_be_passed_pair(self):
        self.assertFalse(self.position_filter.filter_pair('aa bb cc dd ee',
                                                          'aa cc dd ee'))

    def test_empty_input(self):
        self.assertTrue(self.position_filter.filter_pair('ab', ''))
        self.assertTrue(self.position_filter.filter_pair('', ''))


# test PositionFilter.filter_tables method
class FilterTablesTestCases(unittest.TestCase):
    def setUp(self):
        self.tokenizer = create_delimiter_tokenizer(' ')
        self.position_filter = PositionFilter(self.tokenizer,
                                              'JACCARD',
                                              0.8)

    def test_valid_tables(self):
        A = pd.DataFrame([{'id': 1, 'attr':'ab cd ef aa bb'},
                          {'id': 2, 'attr':''},
                          {'id': 3, 'attr':'ab'},
                          {'id': 4, 'attr':'ll oo he'},
                          {'id': 5, 'attr':'xy xx zz fg'}])
        B = pd.DataFrame([{'id': 1, 'attr':'zz fg xx'},
                          {'id': 2, 'attr':'he ll'},
                          {'id': 3, 'attr':'xy pl ou'},
                          {'id': 4, 'attr':'aa bb ab ef'},
                          {'id': 5, 'attr':'fg cd aa ef ab'}])
        expected_pairs = set(['1,4'])
        C = self.position_filter.filter_tables(A, B,
                                               'id', 'id',
                                               'attr', 'attr')
        self.assertEquals(len(C), len(expected_pairs))
        self.assertListEqual(list(C.columns.values),
                             ['_id', 'l_id', 'r_id'])
        actual_pairs = set()
        for idx, row in C.iterrows():
            actual_pairs.add(','.join((str(row['l_id']), str(row['r_id']))))

        self.assertEqual(len(expected_pairs), len(actual_pairs))
        common_pairs = actual_pairs.intersection(expected_pairs)
        self.assertEqual(len(common_pairs),
                         len(expected_pairs))

    def test_empty_tables(self):
        A = pd.DataFrame(columns=['id', 'attr'])
        B = pd.DataFrame(columns=['id', 'attr'])
        C = self.position_filter.filter_tables(A, B,
                                               'id', 'id',
                                               'attr', 'attr')
        self.assertEqual(len(C), 0)
        self.assertListEqual(list(C.columns.values),
                             ['_id', 'l_id', 'r_id'])


# test PositionFilter.filter_candset method
class FilterCandsetTestCases(unittest.TestCase):
    def setUp(self):
        self.tokenizer = create_delimiter_tokenizer(' ')
        self.position_filter = PositionFilter(self.tokenizer,
                                              sim_measure_type='JACCARD',
                                              threshold=0.8)

    def test_valid_candset(self):
        A = pd.DataFrame([{'l_id': 1, 'l_attr':'ab cd ef aa bb'},
                          {'l_id': 2, 'l_attr':''},
                          {'l_id': 3, 'l_attr':'ab'},
                          {'l_id': 4, 'l_attr':'ll oo he'},
                          {'l_id': 5, 'l_attr':'xy xx zz fg'}])
        B = pd.DataFrame([{'r_id': 1, 'r_attr':'zz fg xx'},
                          {'r_id': 2, 'r_attr':'he ll'},
                          {'r_id': 3, 'r_attr':'xy pl ou'},
                          {'r_id': 4, 'r_attr':'aa bb ab ef'},
                          {'r_id': 5, 'r_attr':'fg cd aa ef ab'}])

        # generate cartesian product A x B to be used as candset
        A['tmp_join_key'] = 1
        B['tmp_join_key'] = 1
        C = pd.merge(A[['l_id', 'tmp_join_key']],
                     B[['r_id', 'tmp_join_key']],
                     on='tmp_join_key').drop('tmp_join_key', 1)

        expected_pairs = set(['1,4'])

        D = self.position_filter.filter_candset(C,
                                                'l_id', 'r_id',
                                                A, B,
                                                'l_id', 'r_id',
                                                'l_attr', 'r_attr')
        self.assertEquals(len(D), len(expected_pairs))
        self.assertListEqual(list(D.columns.values),
                             ['l_id', 'r_id'])
        actual_pairs = set()
        for idx, row in D.iterrows():
            actual_pairs.add(','.join((str(row['l_id']), str(row['r_id']))))

        self.assertEqual(len(expected_pairs),
                         len(actual_pairs))
        common_pairs = actual_pairs.intersection(expected_pairs)
        self.assertEqual(len(common_pairs), len(expected_pairs))

    def test_empty_candset(self):
        A = pd.DataFrame(columns=['l_id', 'l_attr'])
        B = pd.DataFrame(columns=['r_id', 'r_attr'])

        # generate cartesian product A x B to be used as candset
        A['tmp_join_key'] = 1
        B['tmp_join_key'] = 1
        C = pd.merge(A[['l_id', 'tmp_join_key']],
                     B[['r_id', 'tmp_join_key']],
                     on='tmp_join_key').drop('tmp_join_key', 1)

        D = self.position_filter.filter_candset(C,
                                                'l_id', 'r_id',
                                                A, B,
                                                'l_id', 'r_id',
                                                'l_attr', 'r_attr')
        self.assertEqual(len(D), 0)
        self.assertListEqual(list(D.columns.values),
                             ['l_id', 'r_id'])
