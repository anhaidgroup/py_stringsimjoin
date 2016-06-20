"""Benchmarks for join methods on synthetic data"""

from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer

from .data_generator import generate_table
from .data_generator import generate_tokens  
from py_stringsimjoin.join.cosine_join import cosine_join
from py_stringsimjoin.join.dice_join import dice_join
from py_stringsimjoin.join.edit_distance_join import edit_distance_join
from py_stringsimjoin.join.jaccard_join import jaccard_join
from py_stringsimjoin.join.overlap_coefficient_join import overlap_coefficient_join
from py_stringsimjoin.join.overlap_join import overlap_join


class SmallJoinBenchmark:
    """Small benchmark 10K x 10K"""
    def setup(self):
        tokens = generate_tokens(6, 2, 5000)
        self.ltable = generate_table(5, 1, tokens,
                                     10000, 'id', 'attr')
        self.rtable = generate_table(5, 1, tokens,
                                     10000, 'id', 'attr')
        self.delim_tok = DelimiterTokenizer(delim_set=[' '], return_set=True)

    def time_jaccard_delim_07(self):
        jaccard_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     self.delim_tok, 0.7)

    def time_cosine_delim_07(self):
        cosine_join(self.ltable, self.rtable,
                    'id', 'id', 'attr', 'attr',
                    self.delim_tok, 0.7)

    def time_overlap_delim_1(self):
        overlap_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     self.delim_tok, 1)


class MediumJoinBenchmark:
    """Medium benchmark 25K x 25K"""
    def setup(self):
        tokens = generate_tokens(6, 2, 5000)
        self.ltable = generate_table(5, 1, tokens,
                                     25000, 'id', 'attr')
        self.rtable = generate_table(5, 1, tokens,
                                     25000, 'id', 'attr')
        self.delim_tok = DelimiterTokenizer(delim_set=[' '], return_set=True)

    def time_jaccard_delim_07(self):
        jaccard_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     self.delim_tok, 0.7)

    def time_cosine_delim_07(self):
        cosine_join(self.ltable, self.rtable,
                    'id', 'id', 'attr', 'attr',
                    self.delim_tok, 0.7)

    def time_overlap_delim_1(self):
        overlap_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     self.delim_tok, 1)


class LargeJoinBenchmark:
    """Large benchmark 50K x 50K"""
    def setup(self):
        tokens = generate_tokens(6, 2, 5000)
        self.ltable = generate_table(5, 1, tokens,
                                     50000, 'id', 'attr')
        self.rtable = generate_table(5, 1, tokens,
                                     50000, 'id', 'attr')
        self.delim_tok = DelimiterTokenizer(delim_set=[' '], return_set=True)

    def time_jaccard_delim_07(self):
        jaccard_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     self.delim_tok, 0.7)

    def time_cosine_delim_07(self):
        cosine_join(self.ltable, self.rtable,
                    'id', 'id', 'attr', 'attr',
                    self.delim_tok, 0.7)

    def time_overlap_delim_1(self):
        overlap_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     self.delim_tok, 1)
