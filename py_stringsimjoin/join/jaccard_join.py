# jaccard join
from joblib import delayed
from joblib import Parallel
import pandas as pd

from py_stringsimjoin.join.set_sim_join import set_sim_join
from py_stringsimjoin.utils.helper_functions import split_table
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_key_attr, validate_input_table, validate_threshold, \
    validate_tokenizer, validate_output_attrs


def jaccard_join(ltable, rtable,
                 l_key_attr, r_key_attr,
                 l_join_attr, r_join_attr,
                 tokenizer,
                 threshold,
                 l_out_attrs=None, r_out_attrs=None,
                 l_out_prefix='l_', r_out_prefix='r_',
                 out_sim_score=True,
                 n_jobs=1):
    """Join two tables using jaccard similarity measure.

    Finds tuple pairs from ltable and rtable such that
    Jaccard(ltable.l_join_attr, rtable.r_join_attr) >= threshold

    Args:
    ltable, rtable : Pandas data frame
    l_key_attr, r_key_attr : String, key attribute from ltable and rtable
    l_join_attr, r_join_attr : String, join attribute from ltable and rtable
    tokenizer : Tokenizer object, tokenizer to be used to tokenize join attributes
    threshold : float, jaccard threshold to be satisfied
    l_out_attrs, r_out_attrs : list of attributes to be included in the output table from ltable and rtable
    l_out_prefix, r_out_prefix : String, prefix to be used in the attribute names of the output table
    out_sim_score : boolean, indicates if similarity score needs to be included in the output table

    Returns:
    result : Pandas data frame
    """
    # check if the input tables are dataframes
    validate_input_table(ltable, 'left table')
    validate_input_table(rtable, 'right table')

    # check if the key attributes and join attributes exist
    validate_attr(l_key_attr, ltable.columns,
                  'key attribute', 'left table')
    validate_attr(r_key_attr, rtable.columns,
                  'key attribute', 'right table')
    validate_attr(l_join_attr, ltable.columns,
                  'join attribute', 'left table')
    validate_attr(r_join_attr, rtable.columns,
                  'join attribute', 'right table')

    # check if the input tokenizer is valid
    validate_tokenizer(tokenizer)

    # check if the input threshold is valid
    validate_threshold(threshold, 'JACCARD')

    # check if the output attributes exist
    validate_output_attrs(l_out_attrs, ltable.columns,
                          r_out_attrs, rtable.columns)

    # check if the key attributes are unique and do not contain missing values
    validate_key_attr(l_key_attr, ltable, 'left table')
    validate_key_attr(r_key_attr, rtable, 'right table')

    if n_jobs == 1:
        output_table = set_sim_join(ltable, rtable,
                                    l_key_attr, r_key_attr,
                                    l_join_attr, r_join_attr,
                                    tokenizer, 'JACCARD', threshold,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score)
        output_table.insert(0, '_id', range(0, len(output_table)))
        return output_table
    else:
        r_splits = split_table(rtable, n_jobs)
        results = Parallel(n_jobs=n_jobs)(delayed(set_sim_join)(
                                              ltable, r_split,
                                              l_key_attr, r_key_attr,
                                              l_join_attr, r_join_attr,
                                              tokenizer,
                                              'JACCARD',
                                              threshold,
                                              l_out_attrs, r_out_attrs,
                                              l_out_prefix, r_out_prefix,
                                              out_sim_score)
                                          for r_split in r_splits)
        output_table = pd.concat(results)
        output_table.insert(0, '_id', range(0, len(output_table)))
        return output_table
