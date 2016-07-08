
import os
import unittest

from nose.tools import assert_equal, raises
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from six import iteritems
import pandas as pd

from py_stringsimjoin.join.overlap_join import overlap_join


class OverlapJoinValidTestCases(unittest.TestCase):

    def test_overlap_join_using_tokenizer_with_return_set_false(self):
        A = pd.DataFrame([{'id':1, 'attr':'hello'}])
        B = pd.DataFrame([{'id':1, 'attr':'he ll'}])
        qg2_tok = QgramTokenizer(2)
        assert_equal(qg2_tok.get_return_set(), False)
        c = overlap_join(A, B, 'id', 'id', 'attr', 'attr', qg2_tok, 1)
        assert_equal(len(c), 1)
        assert_equal(qg2_tok.get_return_set(), False)        
        

class OverlapJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])   
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}]) 
        self.tokenizer = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.threshold = 0.8

    @raises(TypeError)
    def test_overlap_join_invalid_ltable(self):
        overlap_join([], self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_overlap_join_invalid_rtable(self):
        overlap_join(self.A, [], 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_join_invalid_l_key_attr(self):
        overlap_join(self.A, self.B, 'A.invalid_id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_join_invalid_r_key_attr(self):
        overlap_join(self.A, self.B, 'A.id', 'B.invalid_id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_join_invalid_l_join_attr(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.invalid_attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_join_invalid_r_join_attr(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.invalid_attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)                                                     
    def test_overlap_join_numeric_l_join_attr(self):                            
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.int_attr', 'B.attr',
                     self.tokenizer, self.threshold)                            
                                                                                
    @raises(AssertionError)                                                     
    def test_overlap_join_numeric_r_join_attr(self):                            
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.int_attr',
                     self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_overlap_join_invalid_tokenizer(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     [], self.threshold)

    @raises(AssertionError)
    def test_overlap_join_invalid_threshold_below(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, -0.1)

    @raises(AssertionError)
    def test_overlap_join_invalid_threshold_zero(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, 0)

    @raises(AssertionError)
    def test_overlap_join_invalid_comp_op_lt(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '<')

    @raises(AssertionError)
    def test_overlap_join_invalid_comp_op_le(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '<=')

    @raises(AssertionError)
    def test_overlap_join_invalid_l_out_attr(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '>=', True, False,
                     ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_overlap_join_invalid_r_out_attr(self):
        overlap_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '>=', True, False,
                     ['A.attr'], ['B.invalid_attr'])

