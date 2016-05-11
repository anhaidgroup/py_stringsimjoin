from functools import partial
import os
import unittest

from nose.tools import assert_equal
from nose.tools import assert_list_equal
from nose.tools import nottest
from nose.tools import raises
from six import iteritems
import pandas as pd

from py_stringsimjoin.join.join import cosine_join
from py_stringsimjoin.join.join import jaccard_join
from py_stringsimjoin.utils.simfunctions import get_sim_function
from py_stringsimjoin.utils.tokenizers import create_delimiter_tokenizer
from py_stringsimjoin.utils.tokenizers import create_qgram_tokenizer
from py_stringsimjoin.utils.tokenizers import tokenize


JOIN_FN_MAP = {'JACCARD': jaccard_join,
               'COSINE': cosine_join}

DEFAULT_L_OUT_PREFIX = 'l_'
DEFAULT_R_OUT_PREFIX = 'r_'

@nottest
def test_valid_join(scenario, sim_measure_type, args):
    (ltable_path, l_key_attr, l_join_attr) = scenario[0]
    (rtable_path, r_key_attr, r_join_attr) = scenario[1]
    join_fn = JOIN_FN_MAP[sim_measure_type]

    # load input tables for the tests.
    ltable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      ltable_path))
    rtable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      rtable_path))

    # generate cartesian product to be used as candset
    ltable['tmp_join_key'] = 1
    rtable['tmp_join_key'] = 1
    cartprod = pd.merge(ltable[[l_key_attr,
                                l_join_attr,
                                'tmp_join_key']],
                        rtable[[r_key_attr,
                                r_join_attr,
                                'tmp_join_key']],
                        on='tmp_join_key').drop('tmp_join_key', 1)
    ltable.drop('tmp_join_key', 1)
    rtable.drop('tmp_join_key', 1)

    sim_func = get_sim_function(sim_measure_type)

    # apply sim function to the entire cartesian product to obtain
    # the expected set of pairs satisfying the threshold.
    cartprod['sim_score'] = cartprod.apply(lambda row: sim_func(
                tokenize(str(row[l_join_attr]), args[0], sim_measure_type),
                tokenize(str(row[r_join_attr]), args[0], sim_measure_type)),
            axis=1)

    expected_pairs = set()
    for idx, row in cartprod.iterrows():
        if float(row['sim_score']) >= args[1]:
            expected_pairs.add(','.join((str(row[l_key_attr]),
                                         str(row[r_key_attr]))))

    # use join function to obtain actual output pairs.
    actual_candset = join_fn(ltable, rtable,
                             l_key_attr, r_key_attr,
                             l_join_attr, r_join_attr,
                             *args)

    expected_output_attrs = ['_id']
    l_out_prefix = DEFAULT_L_OUT_PREFIX
    r_out_prefix = DEFAULT_R_OUT_PREFIX

    # Check for l_out_prefix in args.
    if len(args) > 4:
        l_out_prefix = args[4]
    expected_output_attrs.append(l_out_prefix + l_key_attr)

    # Check for l_out_attrs in args.
    if len(args) > 2:
        if args[2]:
            for attr in args[2]:
                expected_output_attrs.append(l_out_prefix + attr)

    # Check for r_out_prefix in args.
    if len(args) > 5:
        r_out_prefix = args[5]
    expected_output_attrs.append(r_out_prefix + r_key_attr)

    # Check for r_out_attrs in args.
    if len(args) > 3:
        if args[3]:
            for attr in args[3]:
                expected_output_attrs.append(r_out_prefix + attr)

    # Check for out_sim_score in args. 
    if len(args) > 6:
        if args[6]:
            expected_output_attrs.append('_sim_score')
    else:
        expected_output_attrs.append('_sim_score')

    # verify whether the output table has the necessary attributes.
    assert_list_equal(list(actual_candset.columns.values),
                      expected_output_attrs)

    actual_pairs = set()
    for idx, row in actual_candset.iterrows():
        actual_pairs.add(','.join((str(row[l_out_prefix + l_key_attr]),
                                   str(row[r_out_prefix + r_key_attr]))))

    # verify whether the actual pairs and the expected pairs match.
    assert_equal(len(expected_pairs), len(actual_pairs))
    common_pairs = actual_pairs.intersection(expected_pairs)
    assert_equal(len(common_pairs), len(expected_pairs))

def test_set_sim_join():
    # data to be tested.
    test_scenario_1 = [('data/table_A.csv', 'A.ID', 'A.name'),
                       ('data/table_B.csv', 'B.ID', 'B.name')]
    data = {'TEST_SCENARIO_1' : test_scenario_1}

    # similarity measures to be tested.
    sim_measure_types = ['JACCARD', 'COSINE']

    # similarity thresholds to be tested.
    thresholds = [0.3, 0.5, 0.7, 0.85, 1]

    # tokenizers to be tested.
    tokenizers = {'SPACE_DELIMITER': create_delimiter_tokenizer(),
                  '2_GRAM': create_qgram_tokenizer(),
                  '3_GRAM': create_qgram_tokenizer(3)}

    # Test each combination of similarity measure, threshold and tokenizer
    # for different test scenarios.
    for label, scenario in data.iteritems():
        for sim_measure_type in sim_measure_types:
            for threshold in thresholds:
                for tok_type, tok in tokenizers.iteritems():
                    test_function = partial(test_valid_join, scenario,
                                                             sim_measure_type,
                                                             (tok, threshold))
                    test_function.description = 'Test ' + sim_measure_type + \
                        ' with ' + str(threshold) + ' threshold and ' + \
                        tok_type + ' tokenizer for ' + label + '.'
                    yield test_function,

    # Test each similarity measure with output attributes added.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  1,
                                                  ['A.birth_year', 'A.zipcode'],
                                                  ['B.name', 'B.zipcode']))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with output attributes.'
        yield test_function,

    # Test each similarity measure with a different output prefix.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  1,
                                                  ['A.birth_year', 'A.zipcode'],
                                                  ['B.name', 'B.zipcode'],
                                                  'ltable.', 'rtable.'))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with output attributes and prefix.'
        yield test_function,

    # Test each similarity measure with output_sim_score disabled.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  1,
                                                  ['A.birth_year', 'A.zipcode'],
                                                  ['B.name', 'B.zipcode'],
                                                  'ltable.', 'rtable.',
                                                  False))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with sim_score disabled.'
        yield test_function,

class SetSimJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello'}])
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world'}])
        self.tokenizer = create_delimiter_tokenizer()
        self.threshold = 0.8

    @raises(TypeError)
    def test_jaccard_join_invalid_ltable(self):
        jaccard_join([], self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_jaccard_join_invalid_rtable(self):
        jaccard_join(self.A, [], 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_jaccard_join_invalid_l_key_attr(self):
        jaccard_join(self.A, self.B, 'A.invalid_id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_jaccard_join_invalid_r_key_attr(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.invalid_id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_jaccard_join_invalid_l_join_attr(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.invalid_attr', 'B.attr',
                     self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_jaccard_join_invalid_r_join_attr(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.invalid_attr',
                     self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_jaccard_join_invalid_tokenizer(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     [], self.threshold)

    @raises(AssertionError)
    def test_jaccard_join_invalid_threshold_above(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, 1.5)

    @raises(AssertionError)
    def test_jaccard_join_invalid_threshold_below(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, -0.1)

    @raises(AssertionError)
    def test_jaccard_join_invalid_threshold_zero(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, 0)

    @raises(AssertionError)
    def test_jaccard_join_invalid_l_out_attr(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold,
                     ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_jaccard_join_invalid_r_out_attr(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold,
                     ['A.attr'], ['B.invalid_attr'])

    @raises(TypeError)
    def test_cosine_join_invalid_ltable(self):
        cosine_join([], self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_cosine_join_invalid_rtable(self):
        cosine_join(self.A, [], 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_cosine_join_invalid_l_key_attr(self):
        cosine_join(self.A, self.B, 'A.invalid_id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_cosine_join_invalid_r_key_attr(self):
        cosine_join(self.A, self.B, 'A.id', 'B.invalid_id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_cosine_join_invalid_l_join_attr(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.invalid_attr', 'B.attr',
                    self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_cosine_join_invalid_r_join_attr(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.invalid_attr',
                    self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_cosine_join_invalid_tokenizer(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    [], self.threshold)

    @raises(AssertionError)
    def test_cosine_join_invalid_threshold_above(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, 1.5)

    @raises(AssertionError)
    def test_cosine_join_invalid_threshold_below(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, -0.1)

    @raises(AssertionError)
    def test_cosine_join_invalid_threshold_zero(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, 0)

    @raises(AssertionError)
    def test_cosine_join_invalid_l_out_attr(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold,
                    ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_cosine_join_invalid_r_out_attr(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold,
                    ['A.attr'], ['B.invalid_attr'])
