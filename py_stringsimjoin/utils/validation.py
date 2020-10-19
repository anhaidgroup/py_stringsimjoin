"""Validation utilities"""

import os
import numpy as np
import pandas as pd


from py_stringmatching.tokenizer.tokenizer import Tokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer

from py_stringsimjoin.utils.generic_helper import COMP_OP_MAP


def validate_input_table(table, table_label):
    """Check if the input table is a dataframe."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError(table_label + ' is not a dataframe')
    return True


def validate_attr(attr, table_cols, attr_label, table_label):
    """Check if the attribute exists in the table."""
    if attr not in table_cols:
        raise AssertionError(attr_label + ' \'' + attr + '\' not found in ' + \
                             table_label) 
    return True


def validate_attr_type(attr, attr_type, attr_label, table_label):
    """Check if the attribute is not of numeric type."""
    if attr_type != np.object:
        raise AssertionError(attr_label + ' \'' + attr + '\' in ' + 
                             table_label + ' is not of string type.')
    return True


def validate_key_attr(key_attr, table, table_label):
    """Check if the attribute is a valid key attribute."""
    unique_flag = len(table[key_attr].unique()) == len(table)
    nan_flag = sum(table[key_attr].isnull()) == 0 
    if not (unique_flag and nan_flag):
        raise AssertionError('\'' + key_attr + '\' is not a key attribute ' + \
                             'in ' + table_label)
    return True


def validate_output_attrs(l_out_attrs, l_columns, r_out_attrs, r_columns):
    """Check if the output attributes exist in the original tables."""
    if l_out_attrs:
        for attr in l_out_attrs:
            if attr not in l_columns:
                raise AssertionError('output attribute \'' + attr + \
                                     '\' not found in left table')

    if r_out_attrs:
        for attr in r_out_attrs:
            if attr not in r_columns:
                raise AssertionError('output attribute \'' + attr + \
                                     '\' not found in right table')
    return True


def validate_threshold(threshold, sim_measure_type):
    """Check if the threshold is valid for the sim_measure_type."""
    if sim_measure_type == 'EDIT_DISTANCE':
        if threshold < 0:
            raise AssertionError('threshold for ' + sim_measure_type + \
                                 ' should be greater than or equal to 0')
    elif sim_measure_type == 'OVERLAP':
        if threshold <= 0:
            raise AssertionError('threshold for ' + sim_measure_type + \
                                 ' should be greater than 0')
    else:
        if threshold <= 0 or threshold > 1:
            raise AssertionError('threshold for ' + sim_measure_type + \
                                 ' should be in (0, 1]')
    return True


def validate_tokenizer(tokenizer):
    """Check if the input tokenizer is a valid tokenizer."""
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError('Invalid tokenizer provided as input')
    return True


def validate_tokenizer_for_sim_measure(tokenizer, sim_measure_type):
    """Check if the tokenizer is valid for the similarity measure.
    """
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError('Invalid tokenizer provided as input')

    if sim_measure_type == 'EDIT_DISTANCE':
        if not isinstance(tokenizer, QgramTokenizer):
            raise AssertionError('Invalid tokenizer for EDIT_DISTANCE ' + \
            'measure. Only qgram tokenizer should be used for EDIT_DISTANCE.')

    return True


def validate_sim_measure_type(sim_measure_type):
    """Check if the input sim_measure_type is one of the supported types."""
    sim_measure_types = ['COSINE', 'DICE', 'EDIT_DISTANCE', 'JACCARD',
                         'OVERLAP']
    if sim_measure_type.upper() not in sim_measure_types:
        raise TypeError('\'' + sim_measure_type + '\' is not a valid ' + \
                        'sim_measure_type. Supported types are COSINE, DICE' + \
                        ', EDIT_DISTANCE, JACCARD and OVERLAP.')
    return True


def validate_comp_op_for_sim_measure(comp_op, sim_measure_type):
    """Check if the comparison operator is valid for the sim_measure_type."""
    if sim_measure_type == 'EDIT_DISTANCE':
        if comp_op not in ['<=', '<', '=']:
            raise AssertionError('Comparison operator not supported. ' + \
                'Supported comparison operators for ' + sim_measure_type + \
                ' are <=, < and =.')
    else:
        if comp_op not in ['>=', '>', '=']:
            raise AssertionError('Comparison operator not supported. ' + \
                'Supported comparison operators for ' + sim_measure_type + \
                ' are >=, > and =.')
    return True


def validate_comp_op(comp_op):
    """Check if the comparison operator is valid."""
    if comp_op not in COMP_OP_MAP.keys():
        raise AssertionError('Comparison operator not supported. ' + \
            'Supported comparison operators are >=, >, <=, <, = and !=.')        


def validate_path(path):
    """Check if the given path is valid."""
    if os.path.exists(path) == False:
        raise AssertionError('Invalid path given. Please enter an existing path.')
    return True

### utility functions for disk-based joins
def validate_output_file_path(path):
    """Check if the given output file path is valid."""
    dir = os.path.dirname(os.path.abspath(path))
    return validate_path(dir)

def validate_data_limit(data_limit):
    """Check if the given datalimit is valid."""
    if isinstance(data_limit,int) == False:
        raise AssertionError('data_limit is not an integer')
    if data_limit <= 0:
        raise AssertionError('data_limit should be greater than or equal to 0. We suggest it should be greater than 100K.')
    return True
