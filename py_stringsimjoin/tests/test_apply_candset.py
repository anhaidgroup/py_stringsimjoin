import os
import unittest

from nose.tools import assert_equal, assert_list_equal, raises
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.similarity_measure.jaccard import Jaccard
import pandas as pd

from py_stringsimjoin.utils.candset_utils import apply_candset
from py_stringsimjoin.filter.overlap_filter import OverlapFilter
from py_stringsimjoin.utils.helper_functions import COMP_OP_MAP
from py_stringsimjoin.utils.simfunctions import get_sim_function


DEFAULT_L_OUT_PREFIX = 'l_'
DEFAULT_R_OUT_PREFIX = 'r_'

# define sim function wrapper to convert instance methods into a 
# global function. So that, multiprocessing can pickle the sim function.  
jaccard = Jaccard()
def global_jaccard_func(a, b):
    return jaccard.get_raw_score(a, b)

class ApplyCandsetTestCases(unittest.TestCase):
    def setUp(self):
        ltable_path = os.sep.join(['data', 'table_A.csv'])
        rtable_path = os.sep.join(['data', 'table_B.csv'])
        # load input tables for the tests.
        self.ltable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                          ltable_path))
        self.rtable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                          rtable_path))

        self.l_key_attr = 'A.ID'
        self.r_key_attr = 'B.ID'
        self.l_join_attr = 'A.name'
        self.r_join_attr = 'B.name'

        # remove rows with null value in join attribute 
        self.ltable = self.ltable[pd.notnull(
                          self.ltable[self.l_join_attr])]
        self.rtable = self.rtable[pd.notnull(
                          self.rtable[self.r_join_attr])]

        # remove rows with empty value in join attribute 
        self.ltable = self.ltable[
            self.ltable[self.l_join_attr].apply(len) > 0]
        self.rtable = self.rtable[
            self.rtable[self.r_join_attr].apply(len) > 0]

        # generate cartesian product to be used as candset
        self.ltable['tmp_join_key'] = 1
        self.rtable['tmp_join_key'] = 1
        self.cartprod = pd.merge(self.ltable[[
                                    self.l_key_attr,
                                    self.l_join_attr,
                                    'tmp_join_key']],
                                 self.rtable[[
                                    self.r_key_attr,
                                    self.r_join_attr,
                                    'tmp_join_key']],
                        on='tmp_join_key').drop('tmp_join_key', 1)
        self.ltable.drop('tmp_join_key', 1)
        self.rtable.drop('tmp_join_key', 1)

    def test_apply_candset(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        comp_op = '>='

        # apply sim function to the entire cartesian product to obtain
        # the expected set of pairs satisfying the threshold.
        cartprod = self.cartprod
        cartprod['sim_score'] = cartprod.apply(lambda row: sim_func(
                tok.tokenize(str(row[self.l_join_attr])),
                tok.tokenize(str(row[self.r_join_attr]))),
            axis=1)

        comp_fn = COMP_OP_MAP[comp_op]
        expected_pairs = set()
        for idx, row in cartprod.iterrows():
            if comp_fn(float(row['sim_score']), threshold):
                expected_pairs.add(','.join((str(row[self.l_key_attr]),
                                             str(row[self.r_key_attr]))))

        overlap_filter = OverlapFilter(tok, 1, comp_op)
        candset = overlap_filter.filter_tables(self.ltable, self.rtable,
                              self.l_key_attr, self.r_key_attr,
                              self.l_join_attr, self.r_join_attr)

        output_candset = apply_candset(candset,
            DEFAULT_L_OUT_PREFIX+self.l_key_attr, DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable, self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr, tok, sim_func, threshold, comp_op,
            [self.l_join_attr], [self.r_join_attr], out_sim_score=True)

        expected_output_attrs=['_id',
                               DEFAULT_L_OUT_PREFIX + self.l_key_attr,
                               DEFAULT_L_OUT_PREFIX + self.l_join_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_key_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_join_attr,
                               '_sim_score']

        # verify whether the output table has the necessary attributes.
        assert_list_equal(list(output_candset.columns.values),
                          expected_output_attrs)
        actual_pairs = set()
        for idx, row in output_candset.iterrows():
            actual_pairs.add(','.join((str(row[DEFAULT_L_OUT_PREFIX + self.l_key_attr]),
                                       str(row[DEFAULT_R_OUT_PREFIX + self.r_key_attr]))))

        # verify whether the actual pairs and the expected pairs match.
        assert_equal(len(expected_pairs), len(actual_pairs))
        common_pairs = actual_pairs.intersection(expected_pairs)
        assert_equal(len(common_pairs), len(expected_pairs))

    def test_apply_candset_n_jobs_above_1(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        comp_op = '>='

        # apply sim function to the entire cartesian product to obtain
        # the expected set of pairs satisfying the threshold.
        cartprod = self.cartprod
        cartprod['sim_score'] = cartprod.apply(lambda row: sim_func(
                tok.tokenize(str(row[self.l_join_attr])),
                tok.tokenize(str(row[self.r_join_attr]))),
            axis=1)

        comp_fn = COMP_OP_MAP[comp_op]
        expected_pairs = set()
        for idx, row in cartprod.iterrows():
            if comp_fn(float(row['sim_score']), threshold):
                expected_pairs.add(','.join((str(row[self.l_key_attr]),
                                             str(row[self.r_key_attr]))))

        overlap_filter = OverlapFilter(tok, 1, comp_op)
        candset = overlap_filter.filter_tables(self.ltable, self.rtable,
                              self.l_key_attr, self.r_key_attr,
                              self.l_join_attr, self.r_join_attr)

        output_candset = apply_candset(candset,
            DEFAULT_L_OUT_PREFIX+self.l_key_attr, DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable, self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr, tok, global_jaccard_func, threshold, comp_op,
            [self.l_join_attr], [self.r_join_attr], out_sim_score=True, n_jobs=2)

        expected_output_attrs=['_id',
                               DEFAULT_L_OUT_PREFIX + self.l_key_attr,
                               DEFAULT_L_OUT_PREFIX + self.l_join_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_key_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_join_attr,
                               '_sim_score']

        # verify whether the output table has the necessary attributes.
        assert_list_equal(list(output_candset.columns.values),
                          expected_output_attrs)
        actual_pairs = set()
        for idx, row in output_candset.iterrows():
            actual_pairs.add(','.join((str(row[DEFAULT_L_OUT_PREFIX + self.l_key_attr]),
                                       str(row[DEFAULT_R_OUT_PREFIX + self.r_key_attr]))))

        # verify whether the actual pairs and the expected pairs match.
        assert_equal(len(expected_pairs), len(actual_pairs))
        common_pairs = actual_pairs.intersection(expected_pairs)
        assert_equal(len(common_pairs), len(expected_pairs))

    @raises(TypeError)
    def test_invalid_candset(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset([],
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(AssertionError)
    def test_invalid_candset_l_key_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            'invalid_attr',
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(AssertionError)
    def test_invalid_candset_r_key_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            'invalid_attr',
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(TypeError)
    def test_invalid_ltable(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            [], self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(TypeError)
    def test_invalid_rtable(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, [],
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(AssertionError)
    def test_invalid_l_key_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            'invalid_attr', self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(AssertionError)
    def test_invalid_r_key_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, 'invalid_attr',
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(AssertionError)
    def test_invalid_l_join_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            'invalid_attr', self.r_join_attr,
            tok, sim_func, threshold)

    @raises(AssertionError)
    def test_invalid_r_join_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, 'invalid_attr',
            tok, sim_func, threshold)

    @raises(TypeError)
    def test_invalid_tokenizer(self):
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            sim_func, sim_func, threshold)

    @raises(AssertionError)
    def test_invalid_comp_op(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold, comp_op='ge')

    @raises(AssertionError)
    def test_invalid_l_out_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold, l_out_attrs=['invalid_attr'])

    @raises(AssertionError)
    def test_invalid_r_out_attr(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_candset(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold, r_out_attrs=['invalid_attr'])
