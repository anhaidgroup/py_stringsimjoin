from functools import partial
import os
import unittest

from nose.tools import assert_equal
from nose.tools import assert_list_equal
from nose.tools import nottest
from nose.tools import raises
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from six import iteritems
import pandas as pd

from py_stringsimjoin.join.edit_distance_join import edit_distance_join
from py_stringsimjoin.utils.helper_functions import COMP_OP_MAP
from py_stringsimjoin.utils.simfunctions import get_sim_function


DEFAULT_COMP_OP = '<='
DEFAULT_L_OUT_PREFIX = 'l_'
DEFAULT_R_OUT_PREFIX = 'r_'

@nottest
def test_valid_join(scenario, tok, threshold, comp_op=DEFAULT_COMP_OP, args=()):
    (ltable_path, l_key_attr, l_join_attr) = scenario[0]
    (rtable_path, r_key_attr, r_join_attr) = scenario[1]

    # load input tables for the tests.
    ltable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      ltable_path))
    rtable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      rtable_path))

    # remove rows with null value in join attribute 
    ltable = ltable[pd.notnull(ltable[l_join_attr])]
    rtable = rtable[pd.notnull(rtable[r_join_attr])]

    # remove rows with empty value in join attribute 
    ltable = ltable[ltable[l_join_attr].apply(len) > 0]
    rtable = rtable[rtable[r_join_attr].apply(len) > 0]

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

    sim_measure_type = 'EDIT_DISTANCE'
    sim_func = get_sim_function(sim_measure_type)

    # apply sim function to the entire cartesian product to obtain
    # the expected set of pairs satisfying the threshold.
    cartprod['sim_score'] = cartprod.apply(lambda row: sim_func(
                str(row[l_join_attr]), str(row[r_join_attr])),
            axis=1)

    comp_fn = COMP_OP_MAP[comp_op]

    expected_pairs = set()
    overlap = get_sim_function('OVERLAP')
    for idx, row in cartprod.iterrows():
        l_tokens = tok.tokenize(str(row[l_join_attr]))
        r_tokens = tok.tokenize(str(row[r_join_attr]))

        if len(str(row[l_join_attr])) == 0 or len(str(row[r_join_attr])) == 0:
            continue

        # current edit distance join is approximate. It cannot find matching
        # strings which don't have any common q-grams. Hence, remove pairs
        # that don't have any common q-grams from expected pairs.
        if comp_fn(float(row['sim_score']), threshold):
            if overlap(l_tokens, r_tokens) > 0:
                expected_pairs.add(','.join((str(row[l_key_attr]),
                                             str(row[r_key_attr]))))

    # use join function to obtain actual output pairs.
    actual_candset = edit_distance_join(ltable, rtable,
                                        l_key_attr, r_key_attr,
                                        l_join_attr, r_join_attr,
                                        threshold, comp_op,
                                        *args, tokenizer=tok)

    expected_output_attrs = ['_id']
    l_out_prefix = DEFAULT_L_OUT_PREFIX
    r_out_prefix = DEFAULT_R_OUT_PREFIX

    # Check for l_out_prefix in args.
    if len(args) > 2:
        l_out_prefix = args[2]
    expected_output_attrs.append(l_out_prefix + l_key_attr)

    # Check for l_out_attrs in args.
    if len(args) > 0:
        if args[0]:
            for attr in args[0]:
                expected_output_attrs.append(l_out_prefix + attr)

    # Check for r_out_prefix in args.
    if len(args) > 3:
        r_out_prefix = args[3]
    expected_output_attrs.append(r_out_prefix + r_key_attr)

    # Check for r_out_attrs in args.
    if len(args) > 1:
        if args[1]:
            for attr in args[1]:
                expected_output_attrs.append(r_out_prefix + attr)

    # Check for out_sim_score in args. 
    if len(args) > 4:
        if args[4]:
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

def test_edit_distance_join():
    # data to be tested.
    test_scenario_1 = [('data/table_A.csv', 'A.ID', 'A.name'),
                       ('data/table_B.csv', 'B.ID', 'B.name')]
    data = {'TEST_SCENARIO_1' : test_scenario_1}

    # edit distance thresholds to be tested.
    thresholds = [1, 2, 3, 4, 8, 9, 10]

    # tokenizers to be tested.
    tokenizers = {'2_GRAM': QgramTokenizer(qval=2),
                  '3_GRAM': QgramTokenizer(qval=3)}

    # comparison operators to be tested.
    comp_ops = ['<=', '<', '=']

    sim_measure_type = 'EDIT_DISTANCE'
    # Test each combination of threshold and tokenizer
    # for different test scenarios.
    for label, scenario in iteritems(data):
        for threshold in thresholds:
            for tok_type, tok in iteritems(tokenizers):
                for comp_op in comp_ops:
                    test_function = partial(test_valid_join, scenario, tok,
                                                         threshold, comp_op)
                    test_function.description = 'Test ' + sim_measure_type + \
                        ' with ' + str(threshold) + ' threshold and ' + \
                        tok_type + ' tokenizer for ' + label + '.'
                    yield test_function,

    # Test with output attributes added.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             1, '<=',
                                             (['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode']))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with output attributes.'
    yield test_function,

    # Test with a different output prefix.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             1, '<=',
                                             (['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.'))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with output attributes and prefix.'
    yield test_function,

    # Test with output_sim_score disabled.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             1, '<=',
                                             (['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.',
                                              False))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with sim_score disabled.'
    yield test_function,


class EditDistJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello'}])
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world'}])
        self.threshold = 2
        self.comp_op = '<='

    @raises(TypeError)
    def test_edit_distance_join_invalid_ltable(self):
        edit_distance_join([], self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                           self.threshold)

    @raises(TypeError)
    def test_edit_distance_join_invalid_rtable(self):
        edit_distance_join(self.A, [], 'A.id', 'B.id', 'A.attr', 'B.attr',
                           self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_invalid_l_key_attr(self):
        edit_distance_join(self.A, self.B, 'A.invalid_id', 'B.id',
                           'A.attr', 'B.attr', self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_invalid_r_key_attr(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.invalid_id',
                           'A.attr', 'B.attr', self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_invalid_l_join_attr(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id',
                           'A.invalid_attr', 'B.attr', self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_invalid_r_join_attr(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id',
                           'A.attr', 'B.invalid_attr', self.threshold)

    @raises(TypeError)
    def test_edit_distance_join_invalid_tokenizer(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                           self.threshold, tokenizer=[])

    @raises(AssertionError)
    def test_edit_distance_join_invalid_threshold_below(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                           -0.1)

    @raises(AssertionError)
    def test_edit_distance_join_invalid_comp_op_gt(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                           self.threshold, '>')

    @raises(AssertionError)
    def test_edit_distance_join_invalid_comp_op_ge(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                           self.threshold, '>=')

    @raises(AssertionError)
    def test_edit_distance_join_invalid_l_out_attr(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                           self.threshold, self.comp_op,
                           ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_edit_distance_join_invalid_r_out_attr(self):
        edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                           self.threshold, self.comp_op,
                           ['A.attr'], ['B.invalid_attr'])
