from functools import partial
import os
import unittest
from time import sleep

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

    # Print message that we are in the function
    print('\n\n---------------------------------------------------------------------------')
    print('Entering test_valid_join function.')

    (ltable_path, l_key_attr, l_join_attr) = scenario[0]
    (rtable_path, r_key_attr, r_join_attr) = scenario[1]
    join_fn = JOIN_FN_MAP[sim_measure_type]

    print('Loaded scenario and join_fn.')

    # load input tables for the tests.
    ltable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      ltable_path))
    rtable = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      rtable_path))

    if convert_to_str:
        dataframe_column_to_str(ltable, l_join_attr, inplace=True)
        dataframe_column_to_str(rtable, r_join_attr, inplace=True)

    print('Data tables loaded.')

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

    # Test that name " " is causing error
    print('Testing removing values with " " for name:')
    #ltable_not_missing = ltable_not_missing[ltable_not_missing['A.name'] != ' ']
    #rtable_not_missing = rtable_not_missing[rtable_not_missing['B.name'] != ' ']

    # Print message about if we will use tokenizer
    print('Is len(args) > 3: {}'.format(len(args) > 3))

    if len(args) > 3 and (not args[3]):

        # Print message that we are about to use the tokenizer
        print('About to use tokenizer to obtain ltable_not_missing and rtable_not_missing.')

        ltable_not_missing = ltable_not_missing[ltable_not_missing.apply(
            lambda row: len(args[0].tokenize(str(row[l_join_attr]))), 1) > 0]
        rtable_not_missing = rtable_not_missing[rtable_not_missing.apply(
            lambda row: len(args[0].tokenize(str(row[r_join_attr]))), 1) > 0]

        # Print message that we have finished using the tokenizer here
        print('Finished using tokenizer for ltable_not_missing and rtable_not_missing.')

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

    # Print message that we are about to use the tokenizer
    print('About to use tokenizer to obtain cartprod.')

    # apply sim function to the entire cartesian product to obtain
    # the expected set of pairs satisfying the threshold.
    cartprod['sim_score'] = cartprod.apply(lambda row: round(sim_func(
                args[0].tokenize(str(row[l_join_attr])),
                args[0].tokenize(str(row[r_join_attr]))), 4), axis=1)
    
    # Print message that we have finished using the tokenizer here
    print('Finished using tokenizer for cartprod.')
    print('---------------------------------------------------------------------------\n')
   
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
    tok = DelimiterTokenizer(delim_set=['i'], return_set=True)

    # Test the space delimiter tokenizer
    for sim_measure_type in sim_measure_types:
        test_function = partial(test_valid_join, test_scenario_1, sim_measure_type, (tok, 0.5))
        test_function.description = 'Test Appveyor Error with sim measure {}.'.format(sim_measure_type)
        print('Created test function for sim measure {}'.format(sim_measure_type))
        yield test_function,

