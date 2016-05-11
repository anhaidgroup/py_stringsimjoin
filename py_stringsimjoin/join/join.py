from joblib import delayed
from joblib import Parallel
from six import iteritems
import pandas as pd
import pyprind

from py_stringsimjoin.filter.position_filter import PositionFilter
from py_stringsimjoin.filter.position_filter import _find_candidates as \
                                             find_candidates_position_filter
from py_stringsimjoin.filter.prefix_filter import PrefixFilter
from py_stringsimjoin.filter.prefix_filter import _find_candidates as \
                                             find_candidates_prefix_filter
from py_stringsimjoin.filter.suffix_filter import SuffixFilter
from py_stringsimjoin.filter.filter_utils import get_prefix_length
from py_stringsimjoin.index.position_index import PositionIndex
from py_stringsimjoin.index.prefix_index import PrefixIndex
from py_stringsimjoin.utils.helper_functions import build_dict_from_table
from py_stringsimjoin.utils.helper_functions import \
                                                 find_output_attribute_indices
from py_stringsimjoin.utils.helper_functions import \
                                                 get_output_header_from_tables
from py_stringsimjoin.utils.helper_functions import get_output_row_from_tables
from py_stringsimjoin.utils.helper_functions import split_table
from py_stringsimjoin.utils.simfunctions import get_sim_function
from py_stringsimjoin.utils.tokenizers import tokenize
from py_stringsimjoin.utils.tokenizers import Tokenizer
from py_stringsimjoin.utils.token_ordering import gen_token_ordering_for_tables
from py_stringsimjoin.utils.token_ordering import order_using_token_ordering
from py_stringsimjoin.utils.validation import validate_attr
from py_stringsimjoin.utils.validation import validate_input_table
from py_stringsimjoin.utils.validation import validate_output_attrs
from py_stringsimjoin.utils.validation import validate_threshold
from py_stringsimjoin.utils.validation import validate_tokenizer
from py_stringsimjoin.utils.validation import validate_key_attr


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
        output_table = sim_join(ltable, rtable,
                                l_key_attr, r_key_attr,
                                l_join_attr, r_join_attr,
                                tokenizer,
                                'JACCARD',
                                threshold,
                                l_out_attrs, r_out_attrs,
                                l_out_prefix, r_out_prefix,
                                out_sim_score)
        output_table.insert(0, '_id', range(0, len(output_table)))
        return output_table
    else:
        r_splits = split_table(rtable, n_jobs) 
        results = Parallel(n_jobs=n_jobs)(delayed(sim_join)(ltable, s,
                                             l_key_attr, r_key_attr,
                                             l_join_attr, r_join_attr,
                                             tokenizer,
                                             'JACCARD',
                                             threshold,
                                             l_out_attrs, r_out_attrs,
                                             l_out_prefix, r_out_prefix,
                                             out_sim_score) for s in r_splits)
        output_table = pd.concat(results)
        output_table.insert(0, '_id', range(0, len(output_table)))
        return output_table


def cosine_join(ltable, rtable,
                l_key_attr, r_key_attr,
                l_join_attr, r_join_attr,
                tokenizer,
                threshold,
                l_out_attrs=None, r_out_attrs=None,
                l_out_prefix='l_', r_out_prefix='r_',
                out_sim_score=True,
                n_jobs=1):
    """Join two tables using cosine similarity measure.

    Finds tuple pairs from ltable and rtable such that
    CosineSimilarity(ltable.l_join_attr, rtable.r_join_attr) >= threshold

    Args:
    ltable, rtable : Pandas data frame
    l_key_attr, r_key_attr : String, key attribute from ltable and rtable
    l_join_attr, r_join_attr : String, join attribute from ltable and rtable
    tokenizer : Tokenizer object, tokenizer to be used to tokenize join attributes
    threshold : float, cosine threshold to be satisfied
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
    validate_threshold(threshold, 'COSINE')

    # check if the output attributes exist
    validate_output_attrs(l_out_attrs, ltable.columns,
                          r_out_attrs, rtable.columns)

    # check if the key attributes are unique and do not contain missing values
    validate_key_attr(l_key_attr, ltable, 'left table')
    validate_key_attr(r_key_attr, rtable, 'right table')

    if n_jobs == 1:
        output_table = sim_join(ltable, rtable,
                                l_key_attr, r_key_attr,
                                l_join_attr, r_join_attr,
                                tokenizer,
                                'COSINE',
                                threshold,
                                l_out_attrs, r_out_attrs,
                                l_out_prefix, r_out_prefix,
                                out_sim_score)
        output_table.insert(0, '_id', range(0, len(output_table)))
        return output_table
    else:
        r_splits = split_table(rtable, n_jobs)
        results = Parallel(n_jobs=n_jobs)(delayed(sim_join)(ltable, s,
                                             l_key_attr, r_key_attr,
                                             l_join_attr, r_join_attr,
                                             tokenizer,
                                             'COSINE',
                                             threshold,
                                             l_out_attrs, r_out_attrs,
                                             l_out_prefix, r_out_prefix,
                                             out_sim_score) for s in r_splits)
        output_table = pd.concat(results)
        output_table.insert(0, '_id', range(0, len(output_table)))
        return output_table

def edit_dist_join(ltable, rtable,
                   l_key_attr, r_key_attr,
                   l_join_attr, r_join_attr,
                   tokenizer,
                   threshold,
                   l_out_attrs=None, r_out_attrs=None,
                   l_out_prefix='l_', r_out_prefix='r_',
                   out_sim_score=True):
    """Join two tables using edit distance similarity measure.

    Finds tuple pairs from ltable and rtable such that
    EditDistance(ltable.l_join_attr, rtable.r_join_attr) <= threshold

    Args:
    ltable, rtable : Pandas data frame
    l_key_attr, r_key_attr : String, key attribute from ltable and rtable
    l_join_attr, r_join_attr : String, join attribute from ltable and rtable
    tokenizer : Tokenizer object, tokenizer to be used to tokenize join attributes
    threshold : int, edit distance threshold to be satisfied
    l_out_attrs, r_out_attrs : list of attributes to be included in the output table from ltable and rtable
    l_out_prefix, r_out_prefix : String, prefix to be used in the attribute names of the output table
    out_sim_score : boolean, indicates if edit distance needs to be included in the output table

    Returns:
    result : Pandas data frame
    """
    sim_measure_type = 'EDIT_DISTANCE'
    # find column indices of key attr, join attr and output attrs in ltable
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    l_join_attr_index = l_columns.index(l_join_attr)
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, join attr and output attrs in rtable
    r_columns = list(rtable.columns.values)
    r_key_attr_index = r_columns.index(r_key_attr)
    r_join_attr_index = r_columns.index(r_join_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # build a dictionary on ltable
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index,
                                        l_join_attr_index)

    # build a dictionary on rtable
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index,
                                        r_join_attr_index)

    # generate token ordering using tokens in l_join_attr
    # and r_join_attr
    token_ordering = gen_token_ordering_for_tables(
                         [ltable_dict.values(),
                          rtable_dict.values()],
                         [l_join_attr_index,
                          r_join_attr_index],
                         tokenizer, sim_measure_type)

    # build a dictionary of l_join_attr lengths
    l_join_attr_dict = {}
    for row in ltable_dict.values():
        l_join_attr_dict[row[l_key_attr_index]] = len(str(
                                                      row[l_join_attr_index]))

    # Build prefix index on l_join_attr
    prefix_index = PrefixIndex(ltable_dict.values(),
                               l_key_attr_index, l_join_attr_index,
                               tokenizer, sim_measure_type, threshold,
                               token_ordering)
    prefix_index.build()

    prefix_filter = PrefixFilter(tokenizer, 'EDIT_DISTANCE', threshold)
    sim_fn = get_sim_function('EDIT_DISTANCE')
    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)
    prog_bar = pyprind.ProgBar(len(rtable_dict.keys()))

    for r_row in rtable_dict.values():
        r_id = r_row[r_key_attr_index]
        r_string = str(r_row[r_join_attr_index])
        r_len = len(r_string)
        # check for empty string
        if not r_string:
            continue
        r_join_attr_tokens = tokenize(r_string, tokenizer, sim_measure_type)
        r_ordered_tokens = order_using_token_ordering(r_join_attr_tokens,
                                                      token_ordering)
        candidates = find_candidates_prefix_filter(
                         r_ordered_tokens, len(r_ordered_tokens),
                         prefix_filter, prefix_index) 
        for cand in candidates:
            if r_len - threshold <= l_join_attr_dict[cand] <= r_len + threshold:
                edit_dist = sim_fn(str(ltable_dict[cand][l_join_attr_index]),
                                   r_string)
                if edit_dist <= threshold:
                    if has_output_attributes:
                        output_row = get_output_row_from_tables(
                                         ltable_dict[cand], r_row,
                                         cand, r_id,
                                         l_out_attrs_indices,
                                         r_out_attrs_indices)
                        if out_sim_score:
                            output_row.append(edit_dist)
                        output_rows.append(output_row)
                    else:
                        output_row = [cand, r_id]
                        if out_sim_score:
                            output_row.append(edit_dist)
                        output_rows.append(output_row)

        prog_bar.update()

    output_header = get_output_header_from_tables(
                        l_key_attr, r_key_attr,
                        l_out_attrs, r_out_attrs,
                        l_out_prefix, r_out_prefix)
    if out_sim_score:
        output_header.append("_sim_score")

    # generate a dataframe from the list of output rows
    output_table = pd.DataFrame(output_rows, columns=output_header)
    output_table.insert(0, '_id', range(0, len(output_table)))
    return output_table
    
def sim_join(ltable, rtable,
             l_key_attr, r_key_attr,
             l_join_attr, r_join_attr,
             tokenizer,
             sim_measure_type,
             threshold,
             l_out_attrs=None, r_out_attrs=None,
             l_out_prefix='l_', r_out_prefix='r_',
             out_sim_score=True):
    """Join two tables using a similarity measure.

    Finds tuple pairs from ltable and rtable such that
    sim_measure(ltable.l_join_attr, rtable.r_join_attr) >= threshold

    Args:
    ltable, rtable : Pandas data frame
    l_key_attr, r_key_attr : String, key attribute from ltable and rtable
    l_join_attr, r_join_attr : String, join attribute from ltable and rtable
    tokenizer : Tokenizer object, tokenizer to be used to tokenize join attributes
    sim_measure_type : String, similarity measure type ('JACCARD', 'COSINE', 'DICE', 'OVERLAP')
    threshold : float, similarity threshold to be satisfied
    l_out_attrs, r_out_attrs : list of attributes to be included in the output table from ltable and rtable
    l_out_prefix, r_out_prefix : String, prefix to be used in the attribute names of the output table
    out_sim_score : boolean, indicates if similarity score needs to be included in the output table

    Returns:
    result : Pandas data frame
    """
    # find column indices of key attr, join attr and output attrs in ltable
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    l_join_attr_index = l_columns.index(l_join_attr)
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, join attr and output attrs in rtable
    r_columns = list(rtable.columns.values)
    r_key_attr_index = r_columns.index(r_key_attr)
    r_join_attr_index = r_columns.index(r_join_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # build a dictionary on ltable
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index,
                                        l_join_attr_index)

    # build a dictionary on rtable
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index,
                                        r_join_attr_index)

    # generate token ordering using tokens in l_join_attr
    # and r_join_attr
    token_ordering = gen_token_ordering_for_tables(
                         [ltable_dict.values(),
                          rtable_dict.values()],
                         [l_join_attr_index,
                          r_join_attr_index],
                         tokenizer, sim_measure_type)

    # build a dictionary of tokenized l_join_attr
    l_join_attr_dict = {}
    for row in ltable_dict.values():
        l_join_attr_dict[row[l_key_attr_index]] = order_using_token_ordering(
            tokenize(str(row[l_join_attr_index]), tokenizer, sim_measure_type),
                                                  token_ordering)

    # Build position index on l_join_attr
    position_index = PositionIndex(ltable_dict.values(),
                                   l_key_attr_index, l_join_attr_index,
                                   tokenizer, sim_measure_type,
                                   threshold, token_ordering)
    position_index.build()

    pos_filter = PositionFilter(tokenizer, sim_measure_type, threshold)
    suffix_filter = SuffixFilter(tokenizer, sim_measure_type, threshold)
    sim_fn = get_sim_function(sim_measure_type)
    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)
    prog_bar = pyprind.ProgBar(len(rtable_dict.keys()))

    for r_row in rtable_dict.values():
        r_id = r_row[r_key_attr_index]
        r_string = str(r_row[r_join_attr_index])
        # check for empty string
        if not r_string:
            continue
        r_join_attr_tokens = tokenize(r_string, tokenizer, sim_measure_type)
        r_ordered_tokens = order_using_token_ordering(r_join_attr_tokens,
                                                      token_ordering)
        r_num_tokens = len(r_ordered_tokens)
        r_prefix_length = get_prefix_length(r_num_tokens,
                                            sim_measure_type,
                                            threshold, tokenizer)     

        candidate_overlap = find_candidates_position_filter(
                                r_ordered_tokens, r_num_tokens, r_prefix_length,
                                pos_filter, position_index)
        for cand, overlap in candidate_overlap.iteritems():
            if overlap > 0:
                l_ordered_tokens = l_join_attr_dict[cand]
                l_num_tokens = position_index.get_size(cand)
                l_prefix_length = get_prefix_length(
                                      l_num_tokens,
                                      sim_measure_type,
                                      threshold, tokenizer)
                if not suffix_filter._filter_suffix(
                           l_ordered_tokens[l_prefix_length:],
                           r_ordered_tokens[r_prefix_length:],
                           l_prefix_length,
                           r_prefix_length,
                           l_num_tokens, r_num_tokens):
                    sim_score = sim_fn(l_ordered_tokens, r_ordered_tokens)
                    if sim_score >= threshold:
                        if has_output_attributes:
                            output_row = get_output_row_from_tables(
                                             ltable_dict[cand], r_row,
                                             cand, r_id,
                                             l_out_attrs_indices,
                                             r_out_attrs_indices)
                            if out_sim_score:
                                output_row.append(sim_score)
                            output_rows.append(output_row)
                        else:
                            output_row = [cand, r_id]
                            if out_sim_score:
                                output_row.append(sim_score)
                            output_rows.append(output_row)
        prog_bar.update()

    output_header = get_output_header_from_tables(
                        l_key_attr, r_key_attr,
                        l_out_attrs, r_out_attrs,
                        l_out_prefix, r_out_prefix)
    if out_sim_score:
        output_header.append("_sim_score")

    # generate a dataframe from the list of output rows
    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table
