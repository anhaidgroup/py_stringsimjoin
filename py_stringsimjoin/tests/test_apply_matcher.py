import os
import unittest

from nose.tools import assert_equal, assert_list_equal, raises
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from py_stringmatching.similarity_measure.jaccard import Jaccard
import pandas as pd

from py_stringsimjoin.matcher.apply_matcher import apply_matcher
from py_stringsimjoin.filter.overlap_filter import OverlapFilter
from py_stringsimjoin.utils.converter import dataframe_column_to_str
from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP
from py_stringsimjoin.utils.simfunctions import get_sim_function


DEFAULT_L_OUT_PREFIX = 'l_'
DEFAULT_R_OUT_PREFIX = 'r_'


class ApplyMatcherTestCases(unittest.TestCase):
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

        # convert zipcode column to string
        dataframe_column_to_str(self.ltable, 'A.zipcode', inplace=True)              
        dataframe_column_to_str(self.rtable, 'B.zipcode', inplace=True) 

        # copy of tables without removing any rows with missing value.
        # needed to test allow_missing option.
        self.orig_ltable = self.ltable.copy()
        self.orig_rtable = self.rtable.copy()

        # remove rows with null value in join attribute 
        self.ltable = self.ltable[pd.notnull(
                          self.ltable[self.l_join_attr])]
        self.rtable = self.rtable[pd.notnull(
                          self.rtable[self.r_join_attr])]

        # generate cartesian product to be used as candset
        self.ltable['tmp_join_key'] = 1
        self.rtable['tmp_join_key'] = 1
        self.cartprod = pd.merge(self.ltable[[
                                    self.l_key_attr,
                                    self.l_join_attr,
                                    'A.zipcode',
                                    'tmp_join_key']],
                                 self.rtable[[
                                    self.r_key_attr,
                                    self.r_join_attr,
                                    'B.zipcode',
                                    'tmp_join_key']],
                        on='tmp_join_key').drop('tmp_join_key', 1)
        self.ltable.drop('tmp_join_key', 1)
        self.rtable.drop('tmp_join_key', 1)

    def test_apply_matcher(self):
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
        # compute expected output pairs
        expected_pairs = set()
        for idx, row in cartprod.iterrows():
            if comp_fn(float(row['sim_score']), threshold):
                expected_pairs.add(','.join((str(row[self.l_key_attr]),
                                             str(row[self.r_key_attr]))))

        # use overlap filter to obtain a candset.
        overlap_filter = OverlapFilter(tok, 1, comp_op)
        candset = overlap_filter.filter_tables(self.ltable, self.rtable,
                              self.l_key_attr, self.r_key_attr,
                              self.l_join_attr, self.r_join_attr)

        # apply a jaccard matcher to the candset
        output_candset = apply_matcher(candset,
            DEFAULT_L_OUT_PREFIX+self.l_key_attr, DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable, self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr, tok, sim_func, threshold,
            comp_op, False,
            [self.l_join_attr], [self.r_join_attr], out_sim_score=True)

        expected_output_attrs=['_id',
                               DEFAULT_L_OUT_PREFIX + self.l_key_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_key_attr,
                               DEFAULT_L_OUT_PREFIX + self.l_join_attr,
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

    def test_apply_matcher_n_jobs_above_1(self):
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
        # compute expected output pairs
        expected_pairs = set()
        for idx, row in cartprod.iterrows():
            if comp_fn(float(row['sim_score']), threshold):
                expected_pairs.add(','.join((str(row[self.l_key_attr]),
                                             str(row[self.r_key_attr]))))

        # use overlap filter to obtain a candset.
        overlap_filter = OverlapFilter(tok, 1, comp_op)
        candset = overlap_filter.filter_tables(self.ltable, self.rtable,
                              self.l_key_attr, self.r_key_attr,
                              self.l_join_attr, self.r_join_attr)

        # apply a jaccard matcher to the candset
        output_candset = apply_matcher(candset,
            DEFAULT_L_OUT_PREFIX+self.l_key_attr, DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable, self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr, tok, Jaccard().get_raw_score,
            threshold, comp_op, False,
            [self.l_join_attr], [self.r_join_attr], out_sim_score=True, n_jobs=2)

        expected_output_attrs=['_id',
                               DEFAULT_L_OUT_PREFIX + self.l_key_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_key_attr,
                               DEFAULT_L_OUT_PREFIX + self.l_join_attr,
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

    def test_apply_matcher_with_allow_missing(self):
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

        # compute expected output pairs
        comp_fn = COMP_OP_MAP[comp_op]
        expected_pairs = set()
        for idx, row in cartprod.iterrows():
            if comp_fn(float(row['sim_score']), threshold):
                expected_pairs.add(','.join((str(row[self.l_key_attr]),
                                             str(row[self.r_key_attr]))))

        # find pairs that need to be included in output due to
        # the presence of missing value in one of the join attributes.
        missing_pairs = set()
        for l_idx, l_row in self.orig_ltable.iterrows():
            for r_idx, r_row in self.orig_rtable.iterrows():
                if (pd.isnull(l_row[self.l_join_attr]) or
                    pd.isnull(r_row[self.r_join_attr])):
                    missing_pairs.add(','.join((str(l_row[self.l_key_attr]),
                                                str(r_row[self.r_key_attr]))))

        # add the pairs containing missing value to the set of expected pairs.
        expected_pairs = expected_pairs.union(missing_pairs)

        # use overlap filter to obtain a candset with allow_missing set to True. 
        overlap_filter = OverlapFilter(tok, 1, comp_op, allow_missing=True)
        candset = overlap_filter.filter_tables(self.orig_ltable, self.orig_rtable,
                              self.l_key_attr, self.r_key_attr,
                              self.l_join_attr, self.r_join_attr)

        # apply a jaccard matcher to the candset with allow_missing set to True.
        output_candset = apply_matcher(candset,
            DEFAULT_L_OUT_PREFIX+self.l_key_attr, DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.orig_ltable, self.orig_rtable, self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr, tok, sim_func, threshold,
            comp_op, True, out_sim_score=True)

        expected_output_attrs=['_id',
                               DEFAULT_L_OUT_PREFIX + self.l_key_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_key_attr,
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

    def test_apply_matcher_with_join_attr_of_type_int(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        comp_op = '>='
        l_join_attr = 'A.zipcode'
        r_join_attr = 'B.zipcode'

        # apply sim function to the entire cartesian product to obtain
        # the expected set of pairs satisfying the threshold.
        cartprod = self.cartprod
        cartprod['sim_score'] = cartprod.apply(lambda row: sim_func(
                tok.tokenize(str(row[l_join_attr])),
                tok.tokenize(str(row[r_join_attr]))),
            axis=1)

        comp_fn = COMP_OP_MAP[comp_op]
        # compute expected output pairs
        expected_pairs = set()
        for idx, row in cartprod.iterrows():
            if comp_fn(float(row['sim_score']), threshold):
                expected_pairs.add(','.join((str(row[self.l_key_attr]),
                                             str(row[self.r_key_attr]))))

        # use overlap filter to obtain a candset.
        overlap_filter = OverlapFilter(tok, 1, comp_op)
        candset = overlap_filter.filter_tables(self.ltable, self.rtable,
                              self.l_key_attr, self.r_key_attr,
                              l_join_attr, r_join_attr)

        # apply a jaccard matcher to the candset
        output_candset = apply_matcher(candset,
            DEFAULT_L_OUT_PREFIX+self.l_key_attr, DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable, self.l_key_attr, self.r_key_attr,
            l_join_attr, r_join_attr, tok, sim_func, threshold)

        expected_output_attrs=['_id',
                               DEFAULT_L_OUT_PREFIX + self.l_key_attr,
                               DEFAULT_R_OUT_PREFIX + self.r_key_attr,
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

    def test_empty_candset(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        empty_candset = pd.DataFrame(
                        columns=[DEFAULT_L_OUT_PREFIX+self.l_key_attr,
                                 DEFAULT_R_OUT_PREFIX+self.r_key_attr])
        apply_matcher(empty_candset,
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold)

    @raises(TypeError)
    def test_invalid_candset(self):
        tok = QgramTokenizer(qval=2, return_set=True)
        sim_func = get_sim_function('JACCARD')
        threshold = 0.3
        apply_matcher([],
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
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
        apply_matcher(pd.DataFrame([], columns=['_id', 'l_A.ID', 'r_B.ID']),
            DEFAULT_L_OUT_PREFIX+self.l_key_attr,
            DEFAULT_R_OUT_PREFIX+self.r_key_attr,
            self.ltable, self.rtable,
            self.l_key_attr, self.r_key_attr,
            self.l_join_attr, self.r_join_attr,
            tok, sim_func, threshold, r_out_attrs=['invalid_attr'])
