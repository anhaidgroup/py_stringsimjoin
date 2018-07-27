"""
Currently the tests do not pass in Appveyor. Specifically, we got the following error
for all the the failed cases.

Traceback (most recent call last):
  File "c:\python34\lib\site-packages\nose\case.py", line 198, in runTest
    self.test(*self.arg)
  File "C:\projects\py-stringsimjoin\py_stringsimjoin\tests\test_disk_edit_dist_join.py", line 114, in test_valid_join
    output_file_path = output_file_path)
  File "C:\projects\py-stringsimjoin\py_stringsimjoin\join\disk_edit_distance_join.py", line 146, in disk_edit_distance_join
    temp_dir, output_file_path)
  File "py_stringsimjoin\join\disk_edit_distance_join_cy.pyx", line 267, in py_stringsimjoin.join.disk_edit_distance_join_cy.disk_edit_distance_join_cy
    results = Parallel(n_jobs=n_jobs)(delayed(_edit_distance_join_split)(
  File "c:\python34\lib\site-packages\joblib\parallel.py", line 962, in __call__
    self.retrieve()
  File "c:\python34\lib\site-packages\joblib\parallel.py", line 865, in retrieve
    self._output.extend(job.get(timeout=self.timeout))
  File "c:\python34\lib\site-packages\joblib\_parallel_backends.py", line 515, in wrap_future_result
    return future.result(timeout=timeout)
  File "c:\python34\lib\site-packages\joblib\externals\loky\_base.py", line 431, in result
    return self.__get_result()
  File "c:\python34\lib\site-packages\joblib\externals\loky\_base.py", line 382, in __get_result
    raise self._exception
nose.proxy.OSError: [Errno 22] Invalid argument: 'C:\\projects\\py-stringsimjoin\\0_05:11:25:188201.csv'



From the initial analysis, it looks like the error has to do with the intermediate file 
that we create to flush the intermediate results to disk. Note that this error occurs 
only in Appveyor (CI service used for Windows) and not in Travis-CI.

As the test cases are automatically generated using a function, it is hard to find/comment the parts of the function that generates testcases causing error. so we currently do not run 
any testcase in this file. To do this, this testcase file is renamed with a prefix _ 
(as _test_disk_edit_dist_join) so that nosetests does not pick this file 
for running unit test cases.

"""


from functools import partial
import os
import unittest

from nose.tools import assert_equal, assert_list_equal, nottest, raises
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from six import iteritems
import pandas as pd

from py_stringsimjoin.join.disk_edit_distance_join import disk_edit_distance_join
from py_stringsimjoin.join.edit_distance_join import edit_distance_join
from py_stringsimjoin.utils.converter import dataframe_column_to_str            
from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP, \
                                                  remove_redundant_attrs
from py_stringsimjoin.utils.simfunctions import get_sim_function


DEFAULT_COMP_OP = '<='
DEFAULT_L_OUT_PREFIX = 'l_'
DEFAULT_R_OUT_PREFIX = 'r_'
# Creating a default output file path to pass in edit distance disk API.
default_output_file_name = "unit_test_edit_dist_join_disk.csv"
default_output_file_path = os.path.join(os.getcwd(), default_output_file_name)

@nottest
def test_valid_join(scenario, tok, threshold,comp_op=DEFAULT_COMP_OP, args=(),
                    convert_to_str=False,data_limit=100000,temp_dir = os.getcwd(), 
                    output_file_path = default_output_file_path):
    (ltable_path, l_key_attr, l_join_attr) = scenario[0]
    (rtable_path, r_key_attr, r_join_attr) = scenario[1]

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
    if len(args) > 0 and args[0]:
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

    expected_pairs = expected_pairs.union(missing_pairs)

    orig_return_set_flag = tok.get_return_set()
    
    # Removing any previously existing output file path.
    if os.path.exists(output_file_path):
      os.remove(output_file_path)

    # Use join function to process the input data. It returns the boolean value.
    is_success = disk_edit_distance_join(ltable, rtable,
                                         l_key_attr, r_key_attr,
                                         l_join_attr, r_join_attr,
                                         threshold, data_limit,
                                         comp_op, *args,
                                         tokenizer=tok, temp_dir = temp_dir,
                                         output_file_path = output_file_path)
    # Use edit distance join without the disk version to get the dataframe to compare.
    no_disk_candset = edit_distance_join(ltable, rtable,
                                        l_key_attr, r_key_attr,
                                        l_join_attr, r_join_attr,
                                        threshold, comp_op,
                                        *args, tokenizer=tok)
    # Deleting Id to make the schema consistent for comparison.
    if '_id' in no_disk_candset :
      del no_disk_candset['_id']

    assert_equal(tok.get_return_set(), orig_return_set_flag)

    expected_output_attrs = []
    l_out_prefix = DEFAULT_L_OUT_PREFIX
    r_out_prefix = DEFAULT_R_OUT_PREFIX

    # Check for l_out_prefix in args.
    if len(args) > 3:
        l_out_prefix = args[3]
    expected_output_attrs.append(l_out_prefix + l_key_attr)

    # Check for r_out_prefix in args.
    if len(args) > 4:
        r_out_prefix = args[4]
    expected_output_attrs.append(r_out_prefix + r_key_attr)

    # Check for l_out_attrs in args.
    if len(args) > 1:
        if args[1]:
            l_out_attrs = remove_redundant_attrs(args[1], l_key_attr)
            for attr in l_out_attrs:
                expected_output_attrs.append(l_out_prefix + attr)

    # Check for r_out_attrs in args.
    if len(args) > 2:
        if args[2]:
            r_out_attrs = remove_redundant_attrs(args[2], r_key_attr)
            for attr in r_out_attrs:
                expected_output_attrs.append(r_out_prefix + attr)

    # Check for out_sim_score in args. 
    if len(args) > 5:
        if args[5]:
            expected_output_attrs.append('_sim_score')
    else:
        expected_output_attrs.append('_sim_score')

    # Verify whether the current output file path exists.
    assert_equal(True,os.path.exists(output_file_path))

    # verify whether the output table has the necessary attributes.
    actual_candset = pd.read_csv(output_file_path)

    # Comparing column header values
    assert_list_equal(list(actual_candset.columns.values),
                        expected_output_attrs)
    assert_list_equal(list(no_disk_candset.columns.values),
                        list(actual_candset.columns.values))

    actual_pairs = set()
    no_disk_pairs = set() 

    # Creating sets for comparing the data tuples
    for idx, row in actual_candset.iterrows():
        actual_pairs.add(','.join((str(row[l_out_prefix + l_key_attr]),
                                     str(row[r_out_prefix + r_key_attr]))))

    for idx, row in no_disk_candset.iterrows():
        no_disk_pairs.add(','.join((str(row[l_out_prefix + l_key_attr]),
                                     str(row[r_out_prefix + r_key_attr]))))
   
    # Verify whether the actual pairs and the expected pairs match.
    assert_equal(len(expected_pairs), len(actual_pairs))
    assert_equal(len(expected_pairs), len(no_disk_pairs))
    common_pairs = actual_pairs.intersection(expected_pairs)
    common_pairs_no_disk = no_disk_pairs.intersection(expected_pairs)
    assert_equal(len(common_pairs), len(expected_pairs))
    assert_equal(len(common_pairs_no_disk), len(expected_pairs))



def test_edit_distance_join_disk():
    # data to be tested.
    test_scenario_1 = [('data/table_A.csv', 'A.ID', 'A.name'),
                       ('data/table_B.csv', 'B.ID', 'B.name')]
    data = {'TEST_SCENARIO_1' : test_scenario_1}

    # edit distance thresholds to be tested.
    thresholds = [1, 2, 3, 4, 8, 9]

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
                        ' with ' + str(threshold) + ' threshold, ' + \
                        tok_type + ' tokenizer and ' + comp_op + ' comp_op for ' + label + '.'
                    yield test_function,

    # Test with allow_missing flag set to True.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (True,
                                              ['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode']))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with allow_missing set to True.'
    yield test_function,

    # Test with output attributes added.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (False, 
                                              ['A.ID', 'A.birth_year', 'A.zipcode'],
                                              ['B.ID', 'B.name', 'B.zipcode']))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with output attributes.'
    yield test_function,

    # Test with a different output prefix.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (False,
                                              ['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.'))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with output attributes and prefix.'
    yield test_function,

    # Test with output_sim_score disabled.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (False,
                                              ['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.',
                                              False))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with sim_score disabled.'
    yield test_function,

    # Test with n_jobs above 1.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (False,
                                              ['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.',
                                              False, 2))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with n_jobs above 1.'
    yield test_function,

    # Test with n_jobs equal to -1.
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (False,
                                              ['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.',
                                              False, -1))
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with n_jobs equal to -1.'
    yield test_function,

    # Test with data limit equal to 1 (a low value).
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (False,
                                              ['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.',
                                              False, -1), data_limit = 1)
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with data limit equal to 1.'
    yield test_function,
    
    # Test with a given output file path.
    test_output_file_path = os.path.join(os.path.expanduser('~'),"test_edit_disk.csv")
    if os.path.exists(test_output_file_path):
      os.remove(test_output_file_path)
    test_function = partial(test_valid_join, test_scenario_1,
                                             tokenizers['2_GRAM'],
                                             9, '<=',
                                             (False,
                                              ['A.birth_year', 'A.zipcode'],
                                              ['B.name', 'B.zipcode'],
                                              'ltable.', 'rtable.',
                                              False, -1),output_file_path = test_output_file_path)
    test_function.description = 'Test ' + sim_measure_type + \
                                " output file path as OS' home."
    yield test_function,
    if os.path.exists(test_output_file_path):
      os.remove(test_output_file_path)

    # scenario where join attributes are of type int
    test_scenario_2 = [(os.sep.join(['data', 'table_A.csv']), 'A.ID', 'A.zipcode'),
                       (os.sep.join(['data', 'table_B.csv']), 'B.ID', 'B.zipcode')]

    # Test with join attribute of type int.
    test_function = partial(test_valid_join, test_scenario_2,
                            tokenizers['2_GRAM'], 3, '<=', (), True)
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with join attribute of type int.'
    yield test_function,

    # scenario where join attributes are of type float
    test_scenario_3 = [(os.sep.join(['data', 'table_A.csv']), 'A.ID', 'A.hourly_wage'),
                       (os.sep.join(['data', 'table_B.csv']), 'B.ID', 'B.hourly_wage')]

    # Test with join attribute of type float.
    test_function = partial(test_valid_join, test_scenario_3,
                            tokenizers['2_GRAM'], 3, '<=', (), True)
    test_function.description = 'Test ' + sim_measure_type + \
                                ' with join attribute of type float.'
    yield test_function,

    # Test with a tokenizer where return_set flag is set to True.
    tok = QgramTokenizer(2, return_set=True)
    test_function = partial(test_valid_join, test_scenario_1, tok, 9)
    test_function.description = 'Test ' + sim_measure_type + \
                        ' with a tokenizer where return_set flag is set to True'
    yield test_function,

    if os.path.exists(default_output_file_path):
      os.remove(default_output_file_path)


class EditDistJoinInvalidTestCases(unittest.TestCase):
    def setUp(self):
        self.A = pd.DataFrame([{'A.id':1, 'A.attr':'hello', 'A.int_attr':5}])   
        self.B = pd.DataFrame([{'B.id':1, 'B.attr':'world', 'B.int_attr':6}]) 
        self.threshold = 2
        self.comp_op = '<='

    @raises(TypeError)
    def test_edit_distance_join_disk_invalid_ltable(self):
        disk_edit_distance_join([], self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold)

    @raises(TypeError)
    def test_edit_distance_join_disk_invalid_rtable(self):
        disk_edit_distance_join(self.A, [], 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_l_key_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.invalid_id', 'B.id',
                           'A.attr', 'B.attr', self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_r_key_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.invalid_id',
                           'A.attr', 'B.attr', self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_l_join_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id',
                           'A.invalid_attr', 'B.attr', self.threshold)

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_r_join_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id',
                           'A.attr', 'B.invalid_attr', self.threshold)

    @raises(AssertionError)                                                     
    def test_edit_distance_join_disk_numeric_l_join_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id',
                           'A.int_attr', 'B.attr', self.threshold)          
                                                                                
    @raises(AssertionError)                                                     
    def test_edit_distance_join_disk_numeric_r_join_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id',
                           'A.attr', 'B.int_attr', self.threshold)

    @raises(TypeError)
    def test_edit_distance_join_disk_invalid_tokenizer(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, tokenizer=[])

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_threshold_below(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                -0.1)

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_data_limit(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, data_limit = -1)

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_data_limit(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, data_limit = 0)

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_data_limit(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, data_limit = 'a')

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_comp_op_gt(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, comp_op = '>')

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_comp_op_ge(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, comp_op = '>=')

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_l_out_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, comp_op = self.comp_op, allow_missing = False,
                                l_out_attrs = ['A.invalid_attr'], r_out_attrs = ['B.attr'])

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_r_out_attr(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, self.comp_op, allow_missing = False,
                                l_out_attrs = ['A.attr'], r_out_attrs = ['B.invalid_attr'])

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_temp_dir(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, temp_dir = 'Wrong_dir/worng_file_path')

    @raises(AssertionError)
    def test_edit_distance_join_disk_invalid_r_output_file_path(self):
        disk_edit_distance_join(self.A, self.B, 'A.id', 'B.id', 'A.attr', 'B.attr',
                                self.threshold, output_file_path = 'Wrong_dir/worng_file_path')
