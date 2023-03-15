import unittest

from nose.tools import assert_equal, assert_list_equal, nottest, raises
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
import numpy as np
import pandas as pd

from py_stringsimjoin.filter.size_filter import SizeFilter
from py_stringsimjoin.utils.converter import dataframe_column_to_str            
from py_stringsimjoin.utils.generic_helper import remove_redundant_attrs


# test SizeFilter.filter_pair method
class FilterPairTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.qg2 = QgramTokenizer(2)

    # tests for JACCARD measure
    def test_jac_dlm_08_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy',
                              self.dlm, 'JACCARD', 0.8, False, False, True)

    def test_jac_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy aa tt',
                              self.dlm, 'JACCARD', 0.8, False, False, False)

    # tests for COSINE measure 
    def test_cos_dlm_08_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy',
                              self.dlm, 'COSINE', 0.8, False, False, True)

    def test_cos_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy aa tt',
                              self.dlm, 'COSINE', 0.8, False, False, False)

    # tests for DICE measure 
    def test_dice_dlm_08_prune_lower(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy uu',
                              self.dlm, 'DICE', 0.8, False, False, True)

    def test_dice_dlm_08_prune_upper(self):
        self.test_filter_pair('aa bb cc dd ee', 'cc xx yy aa tt uu ii oo',
                              self.dlm, 'DICE', 0.8, False, False, True)

    def test_dice_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy aa tt',
                              self.dlm, 'DICE', 0.8, False, False, False)

    # tests for OVERLAP measure
    def test_overlap_dlm_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy',
                              self.dlm, 'OVERLAP', 3, False, False, True)

    def test_overlap_dlm_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy aa',
                              self.dlm, 'OVERLAP', 3, False, False, False)

    def test_overlap_dlm_empty(self):
        self.test_filter_pair('', '',
                              self.dlm, 'OVERLAP', 1, False, False, True)

    def test_overlap_dlm_empty_with_allow_empty(self):
        self.test_filter_pair('', '',
                              self.dlm, 'OVERLAP', 1, True, False, True)

    # tests for EDIT_DISTANCE measure
    def test_edit_dist_qg2_prune(self):
        self.test_filter_pair('abcd', 'cd',
                              self.qg2, 'EDIT_DISTANCE', 1, False, False, True)

    def test_edit_dist_qg2_pass(self):
        self.test_filter_pair('abcd', 'cd',
                              self.qg2, 'EDIT_DISTANCE', 2, False, False, False)

    def test_edit_dist_qg2_empty(self):
        self.test_filter_pair('', '',
                              self.qg2, 'EDIT_DISTANCE', 1, False, False, False)

    def test_edit_dist_qg2_empty_with_allow_empty(self):
        self.test_filter_pair('', '',
                              self.qg2, 'EDIT_DISTANCE', 1, True, False, False)

    def test_edit_dist_qg2_no_padding_empty(self):                              
        self.test_filter_pair('', '', QgramTokenizer(2, padding=False),         
                              'EDIT_DISTANCE', 1, False, False, False)

    # test allow_missing flag
    def test_size_filter_pass_missing_left(self):
        self.test_filter_pair(None, 'fg ty',
                              self.dlm, 'DICE', 0.8, False, True, False)

    def test_size_filter_pass_missing_right(self):
        self.test_filter_pair('fg ty', np.NaN,
                              self.dlm, 'DICE', 0.8, False, True, False)

    def test_size_filter_pass_missing_both(self):
        self.test_filter_pair(None, np.NaN,
                              self.dlm, 'DICE', 0.8, False, True, False)

    # tests for empty string input
    def test_empty_lstring(self):
        self.test_filter_pair('ab', '', self.dlm, 'JACCARD', 0.8,
                              False, False, True)

    def test_empty_rstring(self):
        self.test_filter_pair('', 'ab', self.dlm, 'JACCARD', 0.8,
                              False, False, True)

    def test_empty_strings(self):
        self.test_filter_pair('', '', self.dlm, 'JACCARD', 0.8,
                              False, False, True)

    def test_empty_strings_with_allow_empty(self):
        self.test_filter_pair('', '', self.dlm, 'JACCARD', 0.8,
                              True, False, False)

    @nottest
    def test_filter_pair(self, lstring, rstring, tokenizer, sim_measure_type,
                         threshold, allow_empty, allow_missing, expected_output):
        size_filter = SizeFilter(tokenizer, sim_measure_type, threshold,
                                 allow_empty, allow_missing)
        actual_output = size_filter.filter_pair(lstring, rstring)
        assert_equal(actual_output, expected_output)


# test SizeFilter.filter_tables method
class FilterTablesTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.A = pd.DataFrame([{'id': 1, 'attr':'ab cd ef aa bb'},
                               {'id': 2, 'attr':''},
                               {'id': 3, 'attr':'ab'},
                               {'id': 4, 'attr':'ll oo pp'},
                               {'id': 5, 'attr':'xy xx zz fg'},
                               {'id': 6, 'attr':None},
                               {'id': 7, 'attr':''}])
        self.B = pd.DataFrame([{'id': 1, 'attr':'mn'},
                               {'id': 2, 'attr':'he ll'},
                               {'id': 3, 'attr':'xy pl ou'},
                               {'id': 4, 'attr':'aa'},
                               {'id': 5, 'attr':'fg cd aa ef'},
                               {'id': 6, 'attr':np.NaN},
                               {'id': 7, 'attr':' '}])

        self.empty_table = pd.DataFrame(columns=['id', 'attr'])
        self.default_l_out_prefix = 'l_'
        self.default_r_out_prefix = 'r_'

    # tests for JACCARD measure
    def test_jac_dlm_08(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,3', '5,5'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    def test_jac_dlm_08_with_out_attrs(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,3', '5,5'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr']),
                                expected_pairs)

    def test_jac_dlm_08_with_out_prefix(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,3', '5,5'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr'],
                                'ltable.', 'rtable.'),
                                expected_pairs)

    # tests for COSINE measure 
    def test_cos_dlm_08(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,2', '4,3',
                              '4,5', '5,3', '5,5'])
        self.test_filter_tables(self.dlm, 'COSINE', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    # tests for DICE measure 
    def test_dice_dlm_08(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,2', '4,3',
                              '4,5', '5,3', '5,5'])
        self.test_filter_tables(self.dlm, 'DICE', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    # tests for OVERLAP measure 
    def test_overlap_dlm_3(self):
        expected_pairs = set(['1,3', '1,5', '4,3', '4,5', '5,3', '5,5'])
        self.test_filter_tables(self.dlm, 'OVERLAP', 3, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    # tests for EDIT_DISTANCE measure
    def test_edit_distance_qg2_2(self):
        A = pd.DataFrame([{'l_id': 1, 'l_attr':'1990'},
                          {'l_id': 2, 'l_attr':'200'},
                          {'l_id': 3, 'l_attr':'0'},
                          {'l_id': 4, 'l_attr':''},
                          {'l_id': 5, 'l_attr':np.NaN}])
        B = pd.DataFrame([{'r_id': 1, 'r_attr':'200155'},
                          {'r_id': 2, 'r_attr':'19'},
                          {'r_id': 3, 'r_attr':'188'},
                          {'r_id': 4, 'r_attr':''},
                          {'r_id': 5, 'r_attr':np.NaN}])

        qg2_tok = QgramTokenizer(2)
        expected_pairs = set(['1,1', '1,2', '1,3',
                              '2,2', '2,3', '2,3', '3,2', '3,3', '3,4',
                              '4,2', '4,4'])
        self.test_filter_tables(qg2_tok, 'EDIT_DISTANCE', 2, False, False,
                                (A, B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)


    # test allow_missing flag
    def test_jac_dlm_08_allow_missing(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,3', '5,5',
                              '6,1', '6,2', '6,3', '6,4', '6,5',
                              '6,6', '6,7', '1,6', '2,6', '3,6',
                              '4,6', '5,6', '7,6'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, True,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    # test allow_empty flag
    def test_jac_dlm_08_allow_empty(self):
        expected_pairs = set(['1,5', '2,7', '3,1', '3,4', '4,3', '5,5', '7,7'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, True, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    # test allow_empty flag with output attributes
    def test_jac_dlm_08_allow_empty_with_out_attrs(self):
        expected_pairs = set(['1,5', '2,7', '3,1', '3,4', '4,3', '5,5', '7,7'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, True, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr']),
                                expected_pairs)

    # test with n_jobs above 1
    def test_jac_dlm_08_with_njobs_above_1(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,3', '5,5'])
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr'],
                                'ltable.', 'rtable.', 2),
                                expected_pairs)

    # test filter attribute of type int
    def test_jac_qg2_with_filter_attr_of_type_int(self):
        A = pd.DataFrame([{'l_id': 1, 'l_attr':1990},
                          {'l_id': 2, 'l_attr':2000},
                          {'l_id': 3, 'l_attr':0},
                          {'l_id': 4, 'l_attr':-1},
                          {'l_id': 5, 'l_attr':1986}])
        B = pd.DataFrame([{'r_id': 1, 'r_attr':2001},
                          {'r_id': 2, 'r_attr':1992},
                          {'r_id': 3, 'r_attr':1886},
                          {'r_id': 4, 'r_attr':2007},
                          {'r_id': 5, 'r_attr':2012}])

        dataframe_column_to_str(A, 'l_attr', inplace=True)                      
        dataframe_column_to_str(B, 'r_attr', inplace=True) 

        qg2_tok = QgramTokenizer(2, return_set=True)
        expected_pairs = set(['1,1', '1,2', '1,3', '1,4', '1,5',
                              '2,1', '2,2', '2,3', '2,4', '2,5',
                              '5,1', '5,2', '5,3', '5,4', '5,5'])
        self.test_filter_tables(qg2_tok, 'JACCARD', 0.8, False, False,
                                (A, B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for empty table input
    def test_empty_ltable(self):
        expected_pairs = set()
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.empty_table, self.B,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    def test_empty_rtable(self):
        expected_pairs = set()
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.A, self.empty_table,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    def test_empty_tables(self):
        expected_pairs = set()
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.empty_table, self.empty_table,
                                'id', 'id', 'attr', 'attr'),
                                expected_pairs)

    @nottest
    def test_filter_tables(self, tokenizer, sim_measure_type, threshold,
                           allow_empty, allow_missing, args, expected_pairs):
        size_filter = SizeFilter(tokenizer, sim_measure_type, threshold,
                                 allow_empty, allow_missing)
        actual_candset = size_filter.filter_tables(*args)

        expected_output_attrs = ['_id']
        l_out_prefix = self.default_l_out_prefix
        r_out_prefix = self.default_r_out_prefix

        # Check for l_out_prefix in args.
        if len(args) > 8:
            l_out_prefix = args[8]
        expected_output_attrs.append(l_out_prefix + args[2])

        # Check for r_out_prefix in args.
        if len(args) > 9:
            r_out_prefix = args[9]
        expected_output_attrs.append(r_out_prefix + args[3])

        # Check for l_out_attrs in args.
        if len(args) > 6:
            if args[6]:
                l_out_attrs = remove_redundant_attrs(args[6], args[2])
                for attr in l_out_attrs:
                    expected_output_attrs.append(l_out_prefix + attr)

        # Check for r_out_attrs in args.
        if len(args) > 7:
            if args[7]:
                r_out_attrs = remove_redundant_attrs(args[7], args[3])
                for attr in r_out_attrs:
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


# test SizeFilter.filter_candset method
class FilterCandsetTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.A = pd.DataFrame([{'l_id': 1, 'l_attr':'ab cd ef aa bb'},
                               {'l_id': 2, 'l_attr':''},
                               {'l_id': 3, 'l_attr':'ab'},
                               {'l_id': 4, 'l_attr':'ll oo pp'},
                               {'l_id': 5, 'l_attr':'xy xx zz fg'},
                               {'l_id': 6, 'l_attr': np.NaN}])
        self.B = pd.DataFrame([{'r_id': 1, 'r_attr':'mn'},
                               {'r_id': 2, 'r_attr':'he ll'},
                               {'r_id': 3, 'r_attr':'xy pl ou'},
                               {'r_id': 4, 'r_attr':'aa'},
                               {'r_id': 5, 'r_attr':'fg cd aa ef'},
                               {'r_id': 6, 'r_attr':None}])

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
        expected_pairs = set(['1,5', '3,1', '3,4', '4,3', '5,5'])
        self.test_filter_candset(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for COSINE measure
    def test_cos_dlm_08(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,2', '4,3',
                              '4,5', '5,3', '5,5'])
        self.test_filter_candset(self.dlm, 'COSINE', 0.8, False, False,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for DICE measure
    def test_dice_dlm_08(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,2', '4,3',
                              '4,5', '5,3', '5,5'])
        self.test_filter_candset(self.dlm, 'DICE', 0.8, False, False,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # test allow_missing flag
    def test_jac_dlm_08_allow_missing(self):
        expected_pairs = set(['1,5', '3,1', '3,4', '4,3', '5,5',
                              '6,1', '6,2', '6,3', '6,4', '6,5',
                              '6,6', '1,6', '2,6', '3,6', '4,6', '5,6'])
        self.test_filter_candset(self.dlm, 'JACCARD', 0.8, False, True,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for empty candset input
    def test_empty_candset(self):
        expected_pairs = set()
        self.test_filter_candset(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.empty_candset, 'l_id', 'r_id',
                                 self.empty_A, self.empty_B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    @nottest
    def test_filter_candset(self, tokenizer, sim_measure_type, threshold,
                            allow_empty, allow_missing, args, expected_pairs):
        size_filter = SizeFilter(tokenizer, sim_measure_type, threshold,
                                 allow_empty, allow_missing)
        actual_output_candset = size_filter.filter_candset(*args)

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


class SizeFilterInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])   
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}])
        self.tokenizer = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.sim_measure_type = 'JACCARD'
        self.threshold = 0.8

    @raises(TypeError)
    def test_invalid_ltable(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables([], self.B, 'A.id', 'B.id',
                                  'A.attr', 'B.attr')

    @raises(TypeError)
    def test_invalid_rtable(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables(self.A, [], 'A.id', 'B.id',
                                  'A.attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_l_key_attr(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables(self.A, self.B, 'A.invalid_id', 'B.id',
                                  'A.attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_r_key_attr(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables(self.A, self.B, 'A.id', 'B.invalid_id',
                                  'A.attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_l_filter_attr(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                  'A.invalid_attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_r_filter_attr(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                  'A.attr', 'B.invalid_attr')

    @raises(AssertionError)                                                     
    def test_numeric_l_filter_attr(self):                                       
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,         
                                 self.threshold)                                
        size_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',               
                                  'A.int_attr', 'B.attr')                   
                                                                                
    @raises(AssertionError)                                                     
    def test_numeric_r_filter_attr(self):                                       
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,         
                                 self.threshold)                                
        size_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',               
                                  'A.attr', 'B.int_attr')

    @raises(AssertionError)
    def test_invalid_l_out_attr(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                  'A.attr', 'B.attr',
                                  ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_invalid_r_out_attr(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type,
                                 self.threshold)
        size_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                  'A.attr', 'B.attr',
                                  ['A.attr'], ['B.invalid_attr'])

    @raises(TypeError)
    def test_invalid_tokenizer(self):
        size_filter = SizeFilter([], self.sim_measure_type, self.threshold)

    @raises(AssertionError)
    def test_invalid_tokenizer_for_edit_distance(self):
        size_filter = SizeFilter(self.tokenizer, 'EDIT_DISTANCE', 2)

    @raises(TypeError)
    def test_invalid_sim_measure_type(self):
        size_filter = SizeFilter(self.tokenizer, 'INVALID_TYPE', self.threshold)

    @raises(AssertionError)
    def test_invalid_threshold(self):
        size_filter = SizeFilter(self.tokenizer, self.sim_measure_type, 1.2)
