"""Benchmarks for join methods on realworld datasets"""

import os

import pandas as pd

from py_stringsimjoin.join.cosine_join import cosine_join
from py_stringsimjoin.join.dice_join import dice_join
from py_stringsimjoin.join.edit_distance_join import edit_distance_join
from py_stringsimjoin.join.jaccard_join import jaccard_join
from py_stringsimjoin.join.overlap_coefficient_join import overlap_coefficient_join
from py_stringsimjoin.join.overlap_join import overlap_join
from py_stringsimjoin.utils.helper_functions import get_install_path
from py_stringsimjoin.utils.tokenizers import create_qgram_tokenizer, \
                                           create_delimiter_tokenizer


# path where datasets are present
BASE_PATH = os.sep.join([get_install_path(), 'benchmarks', 'example_datasets'])

class RestaurantsJoinBenchmark:
    """Benchmark join methods on restaurants dataset"""

    timeout=600.0

    def setup(self):
        ltable_path = os.sep.join([BASE_PATH, 'restaurants', 'A.csv'])
        rtable_path = os.sep.join([BASE_PATH, 'restaurants', 'B.csv'])

        if not os.path.exists(ltable_path):
            raise NotImplementedError('Left table not found. Skipping benchmark.')

        if not os.path.exists(rtable_path):
            raise NotImplementedError('Right table not found. Skipping benchmark.')

        self.ltable = pd.read_csv(ltable_path)
        self.rtable = pd.read_csv(rtable_path)
        self.l_id_attr = 'ID'
        self.r_id_attr = 'ID'
        self.l_join_attr = 'NAME'
        self.r_join_attr = 'NAME'
        self.delim_tok = create_delimiter_tokenizer()

    def time_jaccard_delim_07(self):
        jaccard_join(self.ltable, self.rtable,
                     self.l_id_attr, self.r_id_attr,
                     self.l_join_attr, self.r_join_attr,
                     self.delim_tok, 0.7)

    def time_cosine_delim_07(self):
        cosine_join(self.ltable, self.rtable,
                    self.l_id_attr, self.r_id_attr,
                    self.l_join_attr, self.r_join_attr,
                    self.delim_tok, 0.7)

    def time_dice_delim_07(self):
        dice_join(self.ltable, self.rtable,
                  self.l_id_attr, self.r_id_attr,
                  self.l_join_attr, self.r_join_attr,
                  self.delim_tok, 0.7)

    def time_overlap_coefficient_delim_07(self):
        overlap_coefficient_join(self.ltable, self.rtable,
                                 self.l_id_attr, self.r_id_attr,
                                 self.l_join_attr, self.r_join_attr,
                                 self.delim_tok, 0.7)

    def time_overlap_delim_1(self):
        overlap_join(self.ltable, self.rtable,
                     self.l_id_attr, self.r_id_attr,
                     self.l_join_attr, self.r_join_attr,
                     self.delim_tok, 1)

    def time_edit_distance_qg2_3(self):
        edit_distance_join(self.ltable, self.rtable,
                           self.l_id_attr, self.r_id_attr,
                           self.l_join_attr, self.r_join_attr, 3)


class MusicJoinBenchmark:
    """Benchmark join methods on music dataset"""

    timeout=600.0

    def setup(self):
        ltable_path = os.sep.join([BASE_PATH, 'music', 'A.csv'])
        rtable_path = os.sep.join([BASE_PATH, 'music', 'B.csv'])

        if not os.path.exists(ltable_path):
            raise NotImplementedError('Left table not found. Skipping benchmark.')

        if not os.path.exists(rtable_path):
            raise NotImplementedError('Right table not found. Skipping benchmark.')

        self.ltable = pd.read_csv(ltable_path)
        self.rtable = pd.read_csv(rtable_path)
        self.l_id_attr = 'Sno'
        self.r_id_attr = 'Sno'
        self.l_join_attr = 'Song_Name'
        self.r_join_attr = 'Song_Name'
        self.delim_tok = create_delimiter_tokenizer()

    def time_jaccard_delim_07(self):
        jaccard_join(self.ltable, self.rtable,
                     self.l_id_attr, self.r_id_attr,
                     self.l_join_attr, self.r_join_attr,
                     self.delim_tok, 0.7)

    def time_cosine_delim_07(self):
        cosine_join(self.ltable, self.rtable,
                    self.l_id_attr, self.r_id_attr,
                    self.l_join_attr, self.r_join_attr,
                    self.delim_tok, 0.7)

    def time_dice_delim_07(self):
        dice_join(self.ltable, self.rtable,
                  self.l_id_attr, self.r_id_attr,
                  self.l_join_attr, self.r_join_attr,
                  self.delim_tok, 0.7)

    def time_overlap_coefficient_delim_07(self):
        overlap_coefficient_join(self.ltable, self.rtable,
                                 self.l_id_attr, self.r_id_attr,
                                 self.l_join_attr, self.r_join_attr,
                                 self.delim_tok, 0.7)

    def time_overlap_delim_1(self):
        overlap_join(self.ltable, self.rtable,
                     self.l_id_attr, self.r_id_attr,
                     self.l_join_attr, self.r_join_attr,
                     self.delim_tok, 1)

    def time_edit_distance_qg2_3(self):
        edit_distance_join(self.ltable, self.rtable,
                           self.l_id_attr, self.r_id_attr,
                           self.l_join_attr, self.r_join_attr, 3)
