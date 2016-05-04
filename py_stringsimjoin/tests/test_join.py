import os
import unittest

from nose.tools import nottest
import pandas as pd

from py_stringsimjoin.join.join import sim_join
from py_stringsimjoin.utils.simfunctions import get_sim_function
from py_stringsimjoin.utils.tokenizers import create_delimiter_tokenizer
from py_stringsimjoin.utils.tokenizers import create_qgram_tokenizer


class JoinTestCases(unittest.TestCase):
    def setUp(self):
        # load input tables for the tests.
        self.ltable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                               'data/table_A.csv'))
        self.rtable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                               'data/table_B.csv'))
        self.l_key_attr = 'A.ID'
        self.r_key_attr = 'B.ID'
        self.l_join_attr = 'A.name'
        self.r_join_attr = 'B.name'

        # generate cartesian product to be used as candset
        self.ltable['tmp_join_key'] = 1
        self.rtable['tmp_join_key'] = 1
        self.cartprod = pd.merge(self.ltable[[self.l_key_attr,
                                              self.l_join_attr,
                                              'tmp_join_key']],
                                 self.rtable[[self.r_key_attr,
                                              self.r_join_attr,
                                              'tmp_join_key']],
                                 on='tmp_join_key').drop('tmp_join_key', 1)
        self.ltable.drop('tmp_join_key', 1)
        self.rtable.drop('tmp_join_key', 1)


@nottest
def create_test_function(sim_measure_type, tokenizer, threshold):

    def test_function(self):
        sim_func = get_sim_function(sim_measure_type)

        # apply sim function to the entire cartesian product to obtain
        # the expected set of pairs satisfying the threshold.
        self.cartprod['sim_score'] = self.cartprod.apply(lambda row: sim_func(
            set(tokenizer(str(row[self.l_join_attr]))),
            set(tokenizer(str(row[self.r_join_attr])))), axis=1)

        expected_pairs = set()
        for idx, row in self.cartprod.iterrows():
            if float(row['sim_score']) >= threshold:
                expected_pairs.add(','.join((str(row[self.l_key_attr]),
                                             str(row[self.r_key_attr]))))

        # use join function to obtain actual output pairs.
        D = sim_join(self.ltable, self.rtable,
                     self.l_key_attr, self.r_key_attr,
                     self.l_join_attr, self.r_join_attr,
                     tokenizer,
                     sim_measure_type,
                     threshold,
                     l_out_prefix='', r_out_prefix='')

        # verify whether the output table has the necessary attributes.
        self.assertListEqual(list(D.columns.values),
                             [self.l_key_attr,
                              self.r_key_attr, '_sim_score'])

        actual_pairs = set()
        for idx, row in D.iterrows():
            actual_pairs.add(','.join((str(row[self.l_key_attr]),
                                       str(row[self.r_key_attr]))))

        # verify whether the actual pairs and the expected pairs match.
        self.assertEqual(len(expected_pairs), len(actual_pairs))
        common_pairs = actual_pairs.intersection(expected_pairs)
        self.assertEqual(len(common_pairs), len(expected_pairs))

    return test_function

# create tokenizers needed for the tests.
delim_tokenizer = create_delimiter_tokenizer(' ')
qg2_tokenizer = create_qgram_tokenizer(2)
qg3_tokenizer = create_qgram_tokenizer(3)

# add join parameters to test. For each set of parameters, a test method
# will be dynamically created.
join_conf = {'jaccard_delim' : ['JACCARD', delim_tokenizer, 0.2],
             'jaccard_qg2' : ['JACCARD', qg2_tokenizer, 0.2],
             'jaccard_qg3' : ['JACCARD', qg3_tokenizer, 0.2],
             'cosine_delim' : ['COSINE', delim_tokenizer, 0.2],
             'cosine_qg2' : ['COSINE', qg2_tokenizer, 0.2],
             'cosine_qg3' : ['COSINE', qg3_tokenizer, 0.2]}

# dynamically create test methods and add it to the test class.
for name, params in join_conf.iteritems():
    test_function = create_test_function(params[0], params[1], params[2])
    setattr(JoinTestCases, 'test_{0}'.format(name), test_function)
    del test_function


