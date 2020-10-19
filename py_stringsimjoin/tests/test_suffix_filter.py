import unittest

from nose.tools import assert_equal, assert_list_equal, nottest, raises
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
import numpy as np
import pandas as pd

from py_stringsimjoin.filter.suffix_filter import SuffixFilter
from py_stringsimjoin.utils.converter import dataframe_column_to_str 
from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP, \
                                                  remove_redundant_attrs
from py_stringsimjoin.utils.simfunctions import get_sim_function


# test SuffixFilter.filter_pair method
class FilterPairTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.qg2 = QgramTokenizer(2)

    # tests for JACCARD measure
    def test_jac_dlm_08_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy cc zz ww',
                              self.dlm, 'JACCARD', 0.8, False, False, True)

    def test_jac_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx aa cc dd ee',
                              self.dlm, 'JACCARD', 0.8, False, False, False)

    # tests for COSINE measure 
    def test_cos_dlm_08_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy cc zz ww',
                              self.dlm, 'COSINE', 0.8, False, False, True)

    def test_cos_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx aa cc dd ee',
                              self.dlm, 'COSINE', 0.8, False, False, False)

    # tests for DICE measure 
    def test_dice_dlm_08_prune(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx yy cc zz ww',
                              self.dlm, 'DICE', 0.8, False, False, True)

    def test_dice_dlm_08_pass(self):
        self.test_filter_pair('aa bb cc dd ee', 'xx aa cc dd ee',
                              self.dlm, 'DICE', 0.8, False, False, False)

    # tests for OVERLAP measure 
    def test_overlap_dlm_2_prune(self):
        self.test_filter_pair('dd ee', 'yy zz',
                              self.dlm, 'OVERLAP', 2, False, False, True)

    def test_overlap_dlm_2_pass(self):
        self.test_filter_pair('dd zz', 'yy zz',
                              self.dlm, 'OVERLAP', 2, False, False, False)

    def test_overlap_dlm_empty(self):
        self.test_filter_pair('', '',
                              self.dlm, 'OVERLAP', 1, False, False, True)

    def test_overlap_dlm_empty_with_allow_empty(self):
        self.test_filter_pair('', '',
                              self.dlm, 'OVERLAP', 1, True, False, True)

    # tests for EDIT_DISTANCE measure
    def test_edit_dist_qg2_prune(self):
        self.test_filter_pair('67126790', '26123485',
                              self.qg2, 'EDIT_DISTANCE', 1, False, False, True)

    def test_edit_dist_qg2_pass(self):
        self.test_filter_pair('128690', '129695',
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
        suffix_filter = SuffixFilter(tokenizer, sim_measure_type, threshold,
                                     allow_empty, allow_missing)
        actual_output = suffix_filter.filter_pair(lstring, rstring)
        assert_equal(actual_output, expected_output)


# test SuffixFilter.filter_tables method
class FilterTablesTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.A = pd.DataFrame([{'id': 1, 'attr':'ab cd ef aa bb'},
                               {'id': 2, 'attr':''},
                               {'id': 3, 'attr':'ab'},
                               {'id': 4, 'attr':'ll oo he'},
                               {'id': 5, 'attr':'xy xx zz fg'},
                               {'id': 6, 'attr':np.NaN},
                               {'id': 7, 'attr':''}])

        self.B = pd.DataFrame([{'id': 1, 'attr':'zz fg xx'},
                               {'id': 2, 'attr':'he ll'},
                               {'id': 3, 'attr':'xy pl ou'},
                               {'id': 4, 'attr':'aa'},
                               {'id': 5, 'attr':'fg cd aa ef ab'},
                               {'id': 6, 'attr':None},
                               {'id': 7, 'attr':' '}])

        self.empty_table = pd.DataFrame(columns=['id', 'attr'])
        self.default_l_out_prefix = 'l_'
        self.default_r_out_prefix = 'r_'

    # tests for JACCARD measure
    def test_jac_dlm_075(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'))

    def test_jac_dlm_075_with_out_attrs(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['id', 'attr'], ['id', 'attr']))

    def test_jac_dlm_075_with_out_prefix(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr'],
                                'ltable.', 'rtable.'))

    # tests for COSINE measure 
    def test_cos_dlm_08(self):
        self.test_filter_tables(self.dlm, 'COSINE', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'))

    # tests for DICE measure 
    def test_dice_dlm_08(self):
        self.test_filter_tables(self.dlm, 'DICE', 0.8, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'))

    # tests for OVERLAP measure 
    def test_overlap_dlm_3(self):
        self.test_filter_tables(self.dlm, 'OVERLAP', 3, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'))

    # tests for EDIT_DISTANCE measure
    def test_edit_distance_qg2_2(self):
        A = pd.DataFrame([{'l_id': 1, 'l_attr':'19990'},
                          {'l_id': 2, 'l_attr':'200'},
                          {'l_id': 3, 'l_attr':'0'},
                          {'l_id': 4, 'l_attr':''},
                          {'l_id': 5, 'l_attr':np.NaN}])
        B = pd.DataFrame([{'r_id': 1, 'r_attr':'200155'},
                          {'r_id': 2, 'r_attr':'190'},
                          {'r_id': 3, 'r_attr':'2010'},
                          {'r_id': 4, 'r_attr':''},
                          {'r_id': 5, 'r_attr':np.NaN},
                          {'r_id': 6, 'r_attr':'18950'}])

        qg2_tok = QgramTokenizer(2)
        expected_pairs = set(['1,2', '1,6', '2,2', '2,3',
                              '3,2', '4,4'])
        self.test_filter_tables(qg2_tok, 'EDIT_DISTANCE', 2, False, False,
                                (A, B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'))

    # test with n_jobs above 1
    def test_jac_dlm_075_with_njobs_above_1(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr'],
                                'ltable.', 'rtable.', 2))

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
        self.test_filter_tables(qg2_tok, 'JACCARD', 0.3, False, False,
                                (A, B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'))

    # test allow_missing flag
    def test_jac_dlm_075_allow_missing(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, False, True,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'))

    # test allow_empty flag
    def test_jac_dlm_075_allow_empty(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, True, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr'))

    # test allow_empty flag with output attributes
    def test_jac_dlm_075_allow_empty_with_out_attrs(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, True, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr']))

    # test with n_jobs above 1
    def test_jac_dlm_075_with_njobs_above_1(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.75, False, False,
                                (self.A, self.B,
                                'id', 'id', 'attr', 'attr',
                                ['attr'], ['attr'],
                                'ltable.', 'rtable.', 2))

    # tests for empty table input
    def test_empty_ltable(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.empty_table, self.B,
                                'id', 'id', 'attr', 'attr'))

    def test_empty_rtable(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.A, self.empty_table,
                                'id', 'id', 'attr', 'attr'))

    def test_empty_tables(self):
        self.test_filter_tables(self.dlm, 'JACCARD', 0.8, False, False,
                                (self.empty_table, self.empty_table,
                                'id', 'id', 'attr', 'attr'))

    @nottest
    def test_filter_tables(self, tokenizer, sim_measure_type, threshold,
                           allow_empty, allow_missing, args):
        suffix_filter = SuffixFilter(tokenizer, sim_measure_type, threshold,
                                     allow_empty, allow_missing)
        
        sim_fn = get_sim_function(sim_measure_type)
        # compute the join output pairs
        join_output_pairs = set()
        for l_idx, l_row in args[0].iterrows():
            for r_idx, r_row in args[1].iterrows():
                # if allow_missing is set to True, then add pairs containing
                # missing value to the join output.
                if pd.isnull(l_row[args[4]]) or pd.isnull(r_row[args[5]]):
                    if allow_missing:
                        join_output_pairs.add(','.join((str(l_row[args[2]]),
                                                        str(r_row[args[3]]))))
                    continue
 
                if sim_measure_type == 'EDIT_DISTANCE':
                    l_join_val = str(l_row[args[4]])
                    r_join_val = str(r_row[args[5]])
                    comp_fn = COMP_OP_MAP['<='] 
                else:
                    l_join_val = tokenizer.tokenize(str(l_row[args[4]]))
                    r_join_val = tokenizer.tokenize(str(r_row[args[5]]))
                    comp_fn = COMP_OP_MAP['>=']

                if (len(l_join_val) == 0 and len(r_join_val) == 0 and 
                    sim_measure_type not in ['OVERLAP', 'EDIT_DISTANCE']):
                    if allow_empty:
                        join_output_pairs.add(','.join((str(l_row[args[2]]),
                                                        str(r_row[args[3]]))))
                    continue

                # if both attributes are not missing and not empty, then check 
                # if the pair satisfies the join condition. If yes, then add it 
                # to the join output.
                if comp_fn(sim_fn(l_join_val, r_join_val), threshold):
                    join_output_pairs.add(','.join((str(l_row[args[2]]),
                                                    str(r_row[args[3]]))))

        
        actual_candset = suffix_filter.filter_tables(*args)

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
            actual_pairs.add(','.join((str(int(row[l_out_prefix + args[2]])),
                                       str(int(row[r_out_prefix + args[3]])))))

        # verify whether all the join output pairs are 
        # present in the actual output pairs
        common_pairs = actual_pairs.intersection(join_output_pairs)
        assert_equal(len(common_pairs), len(join_output_pairs))


# test SuffixFilter.filter_candset method
class FilterCandsetTestCases(unittest.TestCase):
    def setUp(self):
        self.dlm = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.A = pd.DataFrame([{'l_id': 1, 'l_attr':'ab cd ef aa bb'},
                               {'l_id': 2, 'l_attr':''},
                               {'l_id': 3, 'l_attr':'ab'},
                               {'l_id': 4, 'l_attr':'ll oo he'},
                               {'l_id': 5, 'l_attr':'xy xx zz fg'},
                               {'l_id': 6, 'l_attr': np.NaN}])

        self.B = pd.DataFrame([{'r_id': 1, 'r_attr':'zz fg xx'},
                               {'r_id': 2, 'r_attr':'he ll'},
                               {'r_id': 3, 'r_attr':'xy pl ou'},
                               {'r_id': 4, 'r_attr':'aa'},
                               {'r_id': 5, 'r_attr':'fg cd aa ef ab'},
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
    def test_jac_dlm_075(self):
        expected_pairs = set(['1,5', '3,4', '5,1', '5,3'])
        self.test_filter_candset(self.dlm, 'JACCARD', 0.75, False, False,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for COSINE measure
    def test_cos_dlm_08(self):
        expected_pairs = set(['1,5', '3,4', '4,2', '5,1', '5,3'])
        self.test_filter_candset(self.dlm, 'COSINE', 0.8, False, False,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # tests for DICE measure
    def test_dice_dlm_08(self):
        expected_pairs = set(['1,5', '3,4', '4,2', '5,1', '5,3'])
        self.test_filter_candset(self.dlm, 'DICE', 0.8, False, False,
                                (self.C, 'l_id', 'r_id',
                                 self.A, self.B,
                                'l_id', 'r_id', 'l_attr', 'r_attr'),
                                expected_pairs)

    # test allow_missing flag
    def test_jac_dlm_075_allow_missing(self):
        expected_pairs = set(['1,5', '3,4', '5,1', '5,3',
                              '6,1', '6,2', '6,3', '6,4', '6,5',
                              '6,6', '1,6', '2,6', '3,6', '4,6', '5,6'])
        self.test_filter_candset(self.dlm, 'JACCARD', 0.75, False, True,
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
        suffix_filter = SuffixFilter(tokenizer, sim_measure_type, threshold,
                                     allow_empty, allow_missing)
        actual_output_candset = suffix_filter.filter_candset(*args)

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


class SuffixFilterInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])   
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}])
        self.tokenizer = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.sim_measure_type = 'JACCARD'
        self.threshold = 0.8

    @raises(TypeError)
    def test_invalid_ltable(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables([], self.B, 'A.id', 'B.id',
                                    'A.attr', 'B.attr')

    @raises(TypeError)
    def test_invalid_rtable(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables(self.A, [], 'A.id', 'B.id',
                                    'A.attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_l_key_attr(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables(self.A, self.B, 'A.invalid_id', 'B.id',
                                    'A.attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_r_key_attr(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables(self.A, self.B, 'A.id', 'B.invalid_id',
                                    'A.attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_l_filter_attr(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                    'A.invalid_attr', 'B.attr')

    @raises(AssertionError)
    def test_invalid_r_filter_attr(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                    'A.attr', 'B.invalid_attr')

    @raises(AssertionError)                                                     
    def test_numeric_l_filter_attr(self):                                       
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,     
                                     self.threshold)                            
        suffix_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',             
                                    'A.int_attr', 'B.attr')                 
                                                                                
    @raises(AssertionError)                                                     
    def test_numeric_r_filter_attr(self):                                       
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,     
                                     self.threshold)                            
        suffix_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',             
                                    'A.attr', 'B.int_attr')

    @raises(AssertionError)
    def test_invalid_l_out_attr(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                    'A.attr', 'B.attr',
                                    ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_invalid_r_out_attr(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type,
                                     self.threshold)
        suffix_filter.filter_tables(self.A, self.B, 'A.id', 'B.id',
                                    'A.attr', 'B.attr',
                                    ['A.attr'], ['B.invalid_attr'])

    @raises(TypeError)
    def test_invalid_tokenizer(self):
        suffix_filter = SuffixFilter([], self.sim_measure_type, self.threshold)

    @raises(AssertionError)
    def test_invalid_tokenizer_for_edit_distance(self):
        suffix_filter = SuffixFilter(self.tokenizer, 'EDIT_DISTANCE', 2)

    @raises(TypeError)
    def test_invalid_sim_measure_type(self):
        suffix_filter = SuffixFilter(self.tokenizer, 'INVALID_TYPE',
                                     self.threshold)

    @raises(AssertionError)
    def test_invalid_threshold(self):
        suffix_filter = SuffixFilter(self.tokenizer, self.sim_measure_type, 1.2)
