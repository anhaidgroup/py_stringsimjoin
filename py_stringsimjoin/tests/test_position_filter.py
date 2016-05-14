import unittest

from nose.tools import assert_equal
from nose.tools import assert_list_equal
from nose.tools import nottest
import pandas as pd

from py_stringsimjoin.filter.position_filter import PositionFilter
from py_stringsimjoin.utils.tokenizers import create_delimiter_tokenizer


# test PositionFilter.filter_pair method
class FilterPairTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = create_delimiter_tokenizer(' ')

    # tests for JACCARD measure
    def test_jac_dlm_08_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'cc dd',
                              self.dlm, 'JACCARD', 0.8, True)

    def test_jac_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'aa cc dd ee',
                              self.dlm, 'JACCARD', 0.8, False)

    # tests for COSINE measure
    def test_cos_dlm_08_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'cc dd',
                              self.dlm, 'COSINE', 0.8, True)

    def test_cos_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'aa cc dd ee',
                              self.dlm, 'COSINE', 0.8, False)

    # tests for empty string input
    def test_empty_lstring(self):
        self.test_filter_pair('ab', '', self.dlm, 'JACCARD', 0.8, True)

    def test_empty_rstring(self):
        self.test_filter_pair('', 'ab', self.dlm, 'JACCARD', 0.8, True)

    def test_empty_strings(self):
        self.test_filter_pair('', '', self.dlm, 'JACCARD', 0.8, True)


    @nottest
    def test_filter_pair(self, lstring, rstring, tokenizer, sim_measure_type,
                         threshold, expected_output):
        position_filter = PositionFilter(tokenizer, sim_measure_type, threshold)
        actual_output = position_filter.filter_pair(lstring, rstring)
        assert_equal(actual_output, expected_output)


# test PositionFilter.filter_tables method
class FilterTablesTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = create_delimiter_tokenizer(' ')
        self.A = pd.DataFrame([{'id': 1, 'attr':'ab cd ef aa bb'},
                               {'id': 2, 'attr':''},
                               {'id': 3, 'attr':'ab'},
                               {'id': 4, 'attr':'ll oo he'},
                               {'id': 5, 'attr':'xy xx zz fg'}])
        self.B = pd.DataFrame([{'id': 1, 'attr':'zz fg xx'},
                               {'id': 2, 'attr':'he ll'},
                               {'id': 3, 'attr':'xy pl ou'},
                               {'id': 4, 'attr':'aa bb ab ef'},
                               {'id': 5, 'attr':'fg cd aa ef ab'}])
        self.empty_table = pd.DataFrame(columns=['id', 'attr'])
        self.default_l_out_prefix = 'l_'
        self.default_r_out_prefix = 'r_'

    # tests for JACCARD measure
    def test_jac_dlm_08(self):
        expected_pairs = set(['1,4'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    def test_jac_dlm_08_with_out_attrs(self):
        expected_pairs = set(['1,4'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr']),
                                expected_pairs)

    def test_jac_dlm_08_with_out_prefix(self):
        expected_pairs = set(['1,4'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr'],
                                'ltable.', 'rtable.'),
                                expected_pairs)

    # tests for COSINE measure
    def test_cos_dlm_08(self):
        expected_pairs = set(['1,5', '1,4', '4,2', '5,1'])
        self.test_filter_tables(self.dlm, 'COSINE', 0.8,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    def test_cos_dlm_08_with_out_attrs(self):
        expected_pairs = set(['1,5', '1,4', '4,2', '5,1'])
        self.test_filter_tables(self.dlm, 'COSINE', 0.8,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr']),
                                expected_pairs)

    def test_cos_dlm_08_with_out_prefix(self):
        expected_pairs = set(['1,5', '1,4', '4,2', '5,1'])
        self.test_filter_tables(self.dlm, 'COSINE', 0.8,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr'],
                                'ltable.', 'rtable.'),
                                expected_pairs)

    # tests for empty table input
    def test_empty_ltable(self):
        expected_pairs = set()
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8,
                                (self.empty_table, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    def test_empty_rtable(self):
        expected_pairs = set()
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8,
                                (self.A, self.empty_table,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    def test_empty_tables(self):
        expected_pairs = set()
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8,
                                (self.empty_table, self.empty_table,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    @nottest
    def test_filter_tables(self, tokenizer, sim_measure_type, threshold, args,
                           expected_pairs):
        position_filter = PositionFilter(tokenizer, sim_measure_type, threshold)
        actual_candset = position_filter.filter_tables(*args)

        expected_output_attrs = ['_id']
        l_out_prefix = self.default_l_out_prefix
        r_out_prefix = self.default_r_out_prefix

        # Check for l_out_prefix in args.
        if len(args) > 8:
            l_out_prefix = args[8]
        expected_output_attrs.append(l_out_prefix + args[2])

        # Check for l_out_attrs in args.
        if len(args) > 6:
            if args[6]:
                for attr in args[6]:
                    expected_output_attrs.append(l_out_prefix + attr)

        # Check for r_out_prefix in args.
        if len(args) > 9:
            r_out_prefix = args[9]
        expected_output_attrs.append(r_out_prefix + args[3])

        # Check for r_out_attrs in args.
        if len(args) > 7:
            if args[7]:
                for attr in args[7]:
                    expected_output_attrs.append(r_out_prefix + attr)

        # verify whether the output table has the necessary attributes.
        assert_list_equal(list(actual_candset.columns.values),
                          expected_output_attrs)

        actual_pairs = set()
        for idx, row in actual_candset.iterrows():
            actual_pairs.add(','.join((str(row[l_out_prefix + args[2]]),
                                       str(row[r_out_prefix + args[3]]))))

        # verify whether the actual pairs and the expected pairs match.
        assert_equal(len(expected_pairs), len(actual_pairs))
        common_pairs = actual_pairs.intersection(expected_pairs)
        assert_equal(len(common_pairs), len(expected_pairs))


# test PositionFilter.filter_candset method
class FilterCandsetTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = create_delimiter_tokenizer(' ')
        self.A = pd.DataFrame([{'l_id': 1, 'l_attr':'ab cd ef aa bb'},
                               {'l_id': 2, 'l_attr':''},
                               {'l_id': 3, 'l_attr':'ab'},
                               {'l_id': 4, 'l_attr':'ll oo he'},
                               {'l_id': 5, 'l_attr':'xy xx zz fg'}])
        self.B = pd.DataFrame([{'r_id': 1, 'r_attr':'zz fg xx'},
                               {'r_id': 2, 'r_attr':'he ll'},
                               {'r_id': 3, 'r_attr':'xy pl ou'},
                               {'r_id': 4, 'r_attr':'aa bb ab ef'},
                               {'r_id': 5, 'r_attr':'fg cd aa ef ab'}])

        # generate cartesian product A x B to be used as candset
        self.A['tmp_join_key'] = 1
        self.B['tmp_join_key'] = 1
        self.C = pd.merge(self.A[['l_id', 'tmp_join_key']],
                          self.B[['r_id', 'tmp_join_key']],
                     on='tmp_join_key').drop('tmp_join_key', 1)

        self.empty_A = pd.DataFrame(columns=['l_id', 'l_attr'])
        self.empty_B = pd.DataFrame(columns=['r_id', 'r_attr'])
        self.empty_candset = pd.DataFrame(columns=['l_id', 'r_id'])


    # tests for JACCARD measure
    def test_jac_dlm_08(self):
        expected_pairs = set(['1,4'])
        self.test_filter_candset(self.dlm, 'JACCARD', 0.8,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for COSINE measure
    def test_cos_dlm_08(self):
        expected_pairs = set(['1,5', '1,4', '4,2', '5,1'])
        self.test_filter_candset(self.dlm, 'COSINE', 0.8,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for empty candset input
    def test_empty_candset(self):
        expected_pairs = set()
        self.test_filter_candset(self.dlm, 'JACCARD', 0.8,
                                (self.empty_candset, 'l_id', 'r_id',
                                 self.empty_A, self.empty_B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    @nottest
    def test_filter_candset(self, tokenizer, sim_measure_type, threshold, args,
                           expected_pairs):
        position_filter = PositionFilter(tokenizer, sim_measure_type, threshold)
        actual_output_candset = position_filter.filter_candset(*args)

        # verify whether the output table has the necessary attributes.
        assert_list_equal(list(actual_output_candset.columns.values),
                          list(args[0].columns.values))

        actual_pairs = set()
        for idx, row in actual_output_candset.iterrows():
            actual_pairs.add(','.join((str(row[args[1]]), str(row[args[2]]))))

        # verify whether the actual pairs and the expected pairs match.
        assert_equal(len(expected_pairs), len(actual_pairs))
        common_pairs = actual_pairs.intersection(expected_pairs)
        assert_equal(len(common_pairs), len(expected_pairs))
