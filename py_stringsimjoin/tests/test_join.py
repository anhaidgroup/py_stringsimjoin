from functools import partial
import os
import unittest

from nose.tools import assert_equal, assert_list_equal, nottest, raises
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from six import iteritems
import pandas as pd

from py_stringsimjoin.join.cosine_join import cosine_join
from py_stringsimjoin.join.dice_join import dice_join
from py_stringsimjoin.join.jaccard_join import jaccard_join
from py_stringsimjoin.join.overlap_coefficient_join import overlap_coefficient_join
from py_stringsimjoin.join.overlap_join import overlap_join
from py_stringsimjoin.utils.converter import dataframe_column_to_str
from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP, \
                                                  remove_redundant_attrs
from py_stringsimjoin.utils.simfunctions import get_sim_function


JOIN_FN_MAP = {'COSINE': cosine_join,
               'DICE': dice_join,
               'JACCARD': jaccard_join, 
               'OVERLAP_COEFFICIENT': overlap_coefficient_join}

DEFAULT_COMP_OP = '>='
DEFAULT_L_OUT_PREFIX = 'l_'
DEFAULT_R_OUT_PREFIX = 'r_'

@nottest
def test_valid_join(scenario, sim_measure_type, args, convert_to_str=False):
    (ltable_path, l_key_attr, l_join_attr) = scenario[0]
    (rtable_path, r_key_attr, r_join_attr) = scenario[1]
    join_fn = JOIN_FN_MAP[sim_measure_type]

    # load input tables for the tests.
    ltable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      ltable_path))
    rtable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      rtable_path))

    if convert_to_str:
        dataframe_column_to_str(ltable, l_join_attr, inplace=True)
        dataframe_column_to_str(rtable, r_join_attr, inplace=True)

    missing_pairs = set()
    # if allow_missing flag is set, compute missing pairs.
    if len(args) > 4 and args[4]:
        for l_idx, l_row in ltable.iterrows():
            for r_idx, r_row in rtable.iterrows(): 
                if (pd.isnull(l_row[l_join_attr]) or
                    pd.isnull(r_row[r_join_attr])):
                    missing_pairs.add(','.join((str(l_row[l_key_attr]),
                                                str(r_row[r_key_attr]))))

    # remove rows with missing value in join attribute and create new dataframes
    # consisting of rows with non-missing values.
    ltable_not_missing = ltable[pd.notnull(ltable[l_join_attr])].copy()
    rtable_not_missing = rtable[pd.notnull(rtable[r_join_attr])].copy()

    if len(args) > 3 and (not args[3]):
        ltable_not_missing = ltable_not_missing[ltable_not_missing.apply(
            lambda row: len(args[0].tokenize(str(row[l_join_attr]))), 1) > 0]
        rtable_not_missing = rtable_not_missing[rtable_not_missing.apply(
            lambda row: len(args[0].tokenize(str(row[r_join_attr]))), 1) > 0]

    # generate cartesian product to be used as candset
    ltable_not_missing['tmp_join_key'] = 1
    rtable_not_missing['tmp_join_key'] = 1
    cartprod = pd.merge(ltable_not_missing[[l_key_attr,
                                l_join_attr,
                                'tmp_join_key']],
                        rtable_not_missing[[r_key_attr,
                                r_join_attr,
                                'tmp_join_key']],
                        on='tmp_join_key').drop('tmp_join_key', 1)
    ltable_not_missing.drop('tmp_join_key', 1)
    rtable_not_missing.drop('tmp_join_key', 1)

    sim_func = get_sim_function(sim_measure_type)

    # apply sim function to the entire cartesian product to obtain
    # the expected set of pairs satisfying the threshold.
    cartprod['sim_score'] = cartprod.apply(lambda row: round(sim_func(
                args[0].tokenize(str(row[l_join_attr])),
                args[0].tokenize(str(row[r_join_attr]))), 4),
            axis=1)
   
    comp_fn = COMP_OP_MAP[DEFAULT_COMP_OP]
    # Check for comp_op in args.
    if len(args) > 2:
        comp_fn = COMP_OP_MAP[args[2]]

    expected_pairs = set()
    for idx, row in cartprod.iterrows():
        if comp_fn(float(row['sim_score']), args[1]):
            expected_pairs.add(','.join((str(row[l_key_attr]),
                                         str(row[r_key_attr]))))

    expected_pairs = expected_pairs.union(missing_pairs)

    orig_return_set_flag = args[0].get_return_set()

    # use join function to obtain actual output pairs.
    actual_candset = join_fn(ltable, rtable,
                             l_key_attr, r_key_attr,
                             l_join_attr, r_join_attr,
                             *args)

    assert_equal(args[0].get_return_set(), orig_return_set_flag)

    expected_output_attrs = ['_id']
    l_out_prefix = DEFAULT_L_OUT_PREFIX
    r_out_prefix = DEFAULT_R_OUT_PREFIX

    # Check for l_out_prefix in args.
    if len(args) > 7:
        l_out_prefix = args[7]
    expected_output_attrs.append(l_out_prefix + l_key_attr)

    # Check for r_out_prefix in args.
    if len(args) > 8:
        r_out_prefix = args[8]
    expected_output_attrs.append(r_out_prefix + r_key_attr)

    # Check for l_out_attrs in args.
    if len(args) > 5:
        if args[5]:
            l_out_attrs = remove_redundant_attrs(args[5], l_key_attr)
            for attr in l_out_attrs:
                expected_output_attrs.append(l_out_prefix + attr)

    # Check for r_out_attrs in args.
    if len(args) > 6:
        if args[6]:
            r_out_attrs = remove_redundant_attrs(args[6], r_key_attr)
            for attr in r_out_attrs:
                expected_output_attrs.append(r_out_prefix + attr)

    # Check for out_sim_score in args. 
    if len(args) > 9:
        if args[9]:
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
    test_scenario_1 = [(os.sep.join(['data', 'table_A.csv']), 'A.ID', 'A.name'),
                       (os.sep.join(['data', 'table_B.csv']), 'B.ID', 'B.name')]
    data = {'TEST_SCENARIO_1' : test_scenario_1}

    # similarity measures to be tested.
    sim_measure_types = ['COSINE', 'DICE', 'JACCARD', 'OVERLAP_COEFFICIENT']

    # similarity thresholds to be tested.
    thresholds = {'JACCARD' : [0.3, 0.5, 0.7, 0.85, 1],
                  'COSINE' : [0.3, 0.5, 0.7, 0.85, 1], 
                  'DICE' : [0.3, 0.5, 0.7, 0.85, 1],
                  'OVERLAP_COEFFICIENT' : [0.3, 0.5, 0.7, 0.85, 1]}

    # tokenizers to be tested.
    tokenizers = {'SPACE_DELIMITER': DelimiterTokenizer(delim_set=[' '],
                                                        return_set=True),
                  '2_GRAM': QgramTokenizer(qval=2, return_set=True),
                  '3_GRAM': QgramTokenizer(qval=3, return_set=True)}    

    # Test each combination of similarity measure, threshold and tokenizer for different test scenarios.
    for label, scenario in iteritems(data):
        for sim_measure_type in sim_measure_types:
            for threshold in thresholds.get(sim_measure_type):
                for tok_type, tok in iteritems(tokenizers):
                    test_function = partial(test_valid_join, scenario,
                                        sim_measure_type, (tok, threshold))
                    test_function.description = 'Test ' + sim_measure_type + \
                        ' with ' + str(threshold) + ' threshold and ' + \
                        tok_type + ' tokenizer for ' + label + '.'
                    yield test_function,


   # Test each similarity measure with different comparison operators.
    for sim_measure_type in sim_measure_types:
        for comp_op in ['>', '=']:
            test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  0.3, comp_op, False))
            test_function.description = 'Test ' + sim_measure_type + \
                                        ' with comp_op ' + comp_op + '.'
            yield test_function,

    # Test each similarity measure with allow_missing set to True.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  0.7, '>=', False, True))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with allow_missing set to True.'
        yield test_function,

    # Test each similarity measure with output attributes added.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  0.3, '>=', False, False,
                                                  ['A.ID', 'A.birth_year', 'A.zipcode'],
                                                  ['B.ID', 'B.name', 'B.zipcode']))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with output attributes.'
        yield test_function,

    # Test each similarity measure with a different output prefix.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  0.7, '>=', False, False,
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
                                                  0.7, '>=', False, False,
                                                  ['A.birth_year', 'A.zipcode'],
                                                  ['B.name', 'B.zipcode'],
                                                  'ltable.', 'rtable.',
                                                  False))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with sim_score disabled.'
        yield test_function,

    # Test each similarity measure with n_jobs above 1.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  0.3, '>=', False, False,
                                                  ['A.birth_year', 'A.zipcode'],
                                                  ['B.name', 'B.zipcode'],
                                                  'ltable.', 'rtable.',
                                                  False, 2))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with n_jobs above 1.'
        yield test_function,

    # scenario where join attributes are of type int
    test_scenario_2 = [(os.sep.join(['data', 'table_A.csv']), 'A.ID', 'A.zipcode'),
                       (os.sep.join(['data', 'table_B.csv']), 'B.ID', 'B.zipcode')]

   # Test each similarity measure with join attribute of type int.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_2,
                                                 sim_measure_type,
                                                 (tokenizers['2_GRAM'],
                                                  0.3), True)
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with join attribute of type int.'
        yield test_function,

    # scenario where join attributes are of type float
    test_scenario_3 = [(os.sep.join(['data', 'table_A.csv']), 'A.ID', 'A.hourly_wage'),
                       (os.sep.join(['data', 'table_B.csv']), 'B.ID', 'B.hourly_wage')]

   # Test each similarity measure with join attribute of type float.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_3,
                                                 sim_measure_type,
                                                 (tokenizers['2_GRAM'],
                                                  0.3), True)
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with join attribute of type float.'
        yield test_function,

   # Test each similarity measure with a tokenizer with return_set flag set to False.
    for sim_measure_type in sim_measure_types:
        tok = QgramTokenizer(2)
        test_function = partial(test_valid_join, test_scenario_1,
                                             sim_measure_type, (tok, 0.3))
        test_function.description = 'Test ' + sim_measure_type + \
                    ' with a tokenizer with return_set flag set to False .'
        yield test_function,

    # Test each similarity measure with allow_empty set to True.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  0.7, '>=', True))
        test_function.description = 'Test ' + sim_measure_type + \
                                    ' with allow_empty set to True.'
        yield test_function,

    # Test each similarity measure with allow_empty set to True and with output attributes.
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1,
                                                 sim_measure_type,
                                                 (tokenizers['SPACE_DELIMITER'],
                                                  0.7, '>=', True, False,
                                                  ['A.name'], ['B.name']))
        test_function.description = 'Test ' + sim_measure_type + \
                    ' with allow_empty set to True and with output attributes.'
        yield test_function,


class JaccardJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}])
        self.tokenizer = DelimiterTokenizer(delim_set=[' '], return_set=True)
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

    @raises(AssertionError)                                                     
    def test_jaccard_join_numeric_l_join_attr(self):                            
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.int_attr', 'B.attr',
                     self.tokenizer, self.threshold)                            
                                                                                
    @raises(AssertionError)                                                     
    def test_jaccard_join_numeric_r_join_attr(self):                            
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.int_attr',
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
    def test_jaccard_join_invalid_comp_op_lt(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '<')

    @raises(AssertionError)
    def test_jaccard_join_invalid_comp_op_le(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '<=')

    @raises(AssertionError)
    def test_jaccard_join_invalid_l_out_attr(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '>=', True, False,
                     ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_jaccard_join_invalid_r_out_attr(self):
        jaccard_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                     self.tokenizer, self.threshold, '>=', True, False,
                     ['A.attr'], ['B.invalid_attr'])


class CosineJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])   
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}])  
        self.tokenizer = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.threshold = 0.8

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

    @raises(AssertionError)                                                     
    def test_cosine_join_numeric_l_join_attr(self):                            
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.int_attr', 'B.attr',    
                    self.tokenizer, self.threshold)                            
                                                                                
    @raises(AssertionError)                                                     
    def test_cosine_join_numeric_r_join_attr(self):                            
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.int_attr',    
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
    def test_cosine_join_invalid_comp_op_lt(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold, '<')

    @raises(AssertionError)
    def test_cosine_join_invalid_comp_op_le(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold, '<=')

    @raises(AssertionError)
    def test_cosine_join_invalid_l_out_attr(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold, '>=', True, False,
                    ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_cosine_join_invalid_r_out_attr(self):
        cosine_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                    self.tokenizer, self.threshold, '>=', True, False,
                    ['A.attr'], ['B.invalid_attr'])


class DiceJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])   
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}])  
        self.tokenizer = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.threshold = 0.8

    @raises(TypeError)
    def test_dice_join_invalid_ltable(self):
        dice_join([], self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_dice_join_invalid_rtable(self):
        dice_join(self.A, [], 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_dice_join_invalid_l_key_attr(self):
        dice_join(self.A, self.B, 'A.invalid_id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_dice_join_invalid_r_key_attr(self):
        dice_join(self.A, self.B, 'A.id', 'B.invalid_id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_dice_join_invalid_l_join_attr(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.invalid_attr', 'B.attr',
                  self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_dice_join_invalid_r_join_attr(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.invalid_attr',
                  self.tokenizer, self.threshold)

    @raises(AssertionError)                                                     
    def test_dice_join_numeric_l_join_attr(self):                            
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.int_attr', 'B.attr',    
                  self.tokenizer, self.threshold)                            
                                                                                
    @raises(AssertionError)                                                     
    def test_dice_join_numeric_r_join_attr(self):                            
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.int_attr',    
                  self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_dice_join_invalid_tokenizer(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  [], self.threshold)

    @raises(AssertionError)
    def test_dice_join_invalid_threshold_above(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, 1.5)

    @raises(AssertionError)
    def test_dice_join_invalid_threshold_below(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, -0.1)

    @raises(AssertionError)
    def test_dice_join_invalid_threshold_zero(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, 0)

    @raises(AssertionError)
    def test_dice_join_invalid_comp_op_lt(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold, '<')

    @raises(AssertionError)
    def test_dice_join_invalid_comp_op_le(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold, '<=')

    @raises(AssertionError)
    def test_dice_join_invalid_l_out_attr(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold, '>=', True, False,
                  ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_dice_join_invalid_r_out_attr(self):
        dice_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                  self.tokenizer, self.threshold, '>=', True, False,
                  ['A.attr'], ['B.invalid_attr'])


class OverlapCoefficientJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])   
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}])  
        self.tokenizer = DelimiterTokenizer(delim_set=[' '], return_set=True)
        self.threshold = 0.8

    @raises(TypeError)
    def test_overlap_coefficient_join_invalid_ltable(self):
        overlap_coefficient_join([], self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_overlap_coefficient_join_invalid_rtable(self):
        overlap_coefficient_join(self.A, [], 'A.id', 'B.id', 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_l_key_attr(self):
        overlap_coefficient_join(self.A, self.B, 'A.invalid_id', 'B.id',
                                 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_r_key_attr(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.invalid_id',
                                 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_l_join_attr(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.invalid_attr', 'B.attr',
                                 self.tokenizer, self.threshold)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_r_join_attr(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.invalid_attr',
                                 self.tokenizer, self.threshold)

    @raises(AssertionError)                                                     
    def test_overlap_coefficient_join_numeric_l_join_attr(self):                            
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id', 
                                 'A.int_attr', 'B.attr',    
                                 self.tokenizer, self.threshold)                            
                                                                                
    @raises(AssertionError)                                                     
    def test_overlap_coefficient_join_numeric_r_join_attr(self):                            
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id', 
                                 'A.attr', 'B.int_attr',    
                                 self.tokenizer, self.threshold)

    @raises(TypeError)
    def test_overlap_coefficient_join_invalid_tokenizer(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr', [], self.threshold)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_threshold_above(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr', self.tokenizer, 1.5)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_threshold_below(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr', self.tokenizer, -0.1)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_threshold_zero(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr', self.tokenizer, 0)

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_comp_op_lt(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold, '<')

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_comp_op_le(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold, '<=')

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_l_out_attr(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold, '>=',
                                 True, False, ['A.invalid_attr'], ['B.attr'])

    @raises(AssertionError)
    def test_overlap_coefficient_join_invalid_r_out_attr(self):
        overlap_coefficient_join(self.A, self.B, 'A.id', 'B.id',
                                 'A.attr', 'B.attr',
                                 self.tokenizer, self.threshold, '>=',
                                 True, False, ['A.attr'], ['B.invalid_attr'])
