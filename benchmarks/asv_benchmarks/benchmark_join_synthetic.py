"""Benchmarks for join methods on synthetic data"""

from .data_generator import generate_table
from .data_generator import generate_tokens  
from py_stringsimjoin.join.cosine_join import cosine_join
from py_stringsimjoin.join.dice_join import dice_join
from py_stringsimjoin.join.edit_distance_join import edit_distance_join
from py_stringsimjoin.join.jaccard_join import jaccard_join
from py_stringsimjoin.join.overlap_coefficient_join import overlap_coefficient_join
from py_stringsimjoin.join.overlap_join import overlap_join
from py_stringsimjoin.utils.tokenizers import create_qgram_tokenizer, \
                                           create_delimiter_tokenizer


class SmallJoinBenchmark:
    """Small benchmark 10K x 10K"""
    def setup(self):
        tokens = generate_tokens(6, 2, 5000)
        self.ltable = generate_table(5, 1, tokens,
                                     10000, 'id', 'attr')
        self.rtable = generate_table(5, 1, tokens,
                                     10000, 'id', 'attr')

    def time_jaccard_delim_07(self):
        dl = create_delimiter_tokenizer()
        jaccard_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     dl, 0.7)

    def time_cosine_delim_07(self):
        dl = create_delimiter_tokenizer()
        cosine_join(self.ltable, self.rtable,
                    'id', 'id', 'attr', 'attr',
                    dl, 0.7)

    def time_overlap_delim_1(self):
        dl = create_delimiter_tokenizer()
        overlap_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     dl, 1)


class MediumJoinBenchmark:
    """Medium benchmark 25K x 25K"""
    def setup(self):
        tokens = generate_tokens(6, 2, 5000)
        self.ltable = generate_table(5, 1, tokens,
                                     25000, 'id', 'attr')
        self.rtable = generate_table(5, 1, tokens,
                                     25000, 'id', 'attr')

    def time_jaccard_delim_07(self):
        dl = create_delimiter_tokenizer()
        jaccard_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     dl, 0.7)

    def time_cosine_delim_07(self):
        dl = create_delimiter_tokenizer()
        cosine_join(self.ltable, self.rtable,
                    'id', 'id', 'attr', 'attr',
                    dl, 0.7)

    def time_overlap_delim_1(self):
        dl = create_delimiter_tokenizer()
        overlap_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     dl, 1)


class LargeJoinBenchmark:
    """Large benchmark 50K x 50K"""
    def setup(self):
        tokens = generate_tokens(6, 2, 5000)
        self.ltable = generate_table(5, 1, tokens,
                                     50000, 'id', 'attr')
        self.rtable = generate_table(5, 1, tokens,
                                     50000, 'id', 'attr')

    def time_jaccard_delim_07(self):
        dl = create_delimiter_tokenizer()
        jaccard_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     dl, 0.7)

    def time_cosine_delim_07(self):
        dl = create_delimiter_tokenizer()
        cosine_join(self.ltable, self.rtable,
                    'id', 'id', 'attr', 'attr',
                    dl, 0.7)

    def time_overlap_delim_1(self):
        dl = create_delimiter_tokenizer()
        overlap_join(self.ltable, self.rtable,
                     'id', 'id', 'attr', 'attr',
                     dl, 1)
