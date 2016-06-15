# overlap coefficient join
from joblib import delayed
from joblib import Parallel
from six import iteritems
import pandas as pd
import pyprind

from py_stringsimjoin.filter.overlap_filter import _find_candidates
from py_stringsimjoin.index.inverted_index import InvertedIndex
from py_stringsimjoin.utils.helper_functions import build_dict_from_table, \
    find_output_attribute_indices, get_output_header_from_tables, \
    get_output_row_from_tables, split_table
from py_stringsimjoin.utils.tokenizers import tokenize
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_key_attr, validate_input_table, validate_threshold, \
    validate_tokenizer, validate_output_attrs


def overlap_coefficient_join(ltable, rtable,
                             l_key_attr, r_key_attr,
                             l_join_attr, r_join_attr,
                             tokenizer,
                             threshold,
                             l_out_attrs=None, r_out_attrs=None,
                             l_out_prefix='l_', r_out_prefix='r_',
                             out_sim_score=True, n_jobs=1):
    """Join two tables using overlap coefficient.

    Finds tuple pairs from ltable and rtable such that
    OverlapCoefficient(ltable.l_join_attr, rtable.r_join_attr) >= threshold

    Args:
        ltable (dataframe): left input table.
        rtable (dataframe): right input table.
        l_key_attr (string): key attribute in left table.
        r_key_attr (string): key attribute in right table.
        l_join_attr (string): join attribute in left table.
        r_join_attr (string): join attribute in right table.
        tokenizer (Tokenizer object): tokenizer to be used to tokenize join attributes.
        threshold (float): overlap coefficient threshold to be satisfied.
        l_out_attrs (list): list of attributes to be included in the output table from
                            left table (defaults to None).
        r_out_attrs (list): list of attributes to be included in the output table from
                            right table (defaults to None).
        l_out_prefix (string): prefix to use for the attribute names coming from left
                               table (defaults to 'l_').
        r_out_prefix (string): prefix to use for the attribute names coming from right
                               table (defaults to 'r_').
        out_sim_score (boolean): flag to indicate if similarity score needs to be
                                 included in the output table (defaults to True).

    Returns:
        output table (dataframe)
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
    validate_threshold(threshold, 'OVERLAP_COEFFICIENT')

    # check if the output attributes exist
    validate_output_attrs(l_out_attrs, ltable.columns,
                          r_out_attrs, rtable.columns)

    # check if the key attributes are unique and do not contain missing values
    validate_key_attr(l_key_attr, ltable, 'left table')
    validate_key_attr(r_key_attr, rtable, 'right table')

    if n_jobs == 1:
        output_table = _overlap_coefficient_join_split(ltable, rtable,
                                           l_key_attr, r_key_attr,
                                           l_join_attr, r_join_attr,
                                           tokenizer,
                                           threshold,
                                           l_out_attrs, r_out_attrs,
                                           l_out_prefix, r_out_prefix,
                                           out_sim_score)
    else:
        r_splits = split_table(rtable, n_jobs)
        results = Parallel(n_jobs=n_jobs)(delayed(_overlap_coefficient_join_split)(
                                              ltable, r_split,
                                              l_key_attr, r_key_attr,
                                              l_join_attr, r_join_attr,
                                              tokenizer,
                                              threshold,
                                              l_out_attrs, r_out_attrs,
                                              l_out_prefix, r_out_prefix,
                                              out_sim_score)
                                          for r_split in r_splits)
        output_table = pd.concat(results)

    output_table.insert(0, '_id', range(0, len(output_table)))
    return output_table


def _overlap_coefficient_join_split(ltable, rtable,
                                    l_key_attr, r_key_attr,
                                    l_join_attr, r_join_attr,
                                    tokenizer,
                                    threshold,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score):
    """Perform overlap coefficient join for a split of ltable and rtable"""
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

    # Build inverted index over ltable
    inverted_index = InvertedIndex(ltable_dict.values(),
                                   l_key_attr_index, l_join_attr_index,
                                   tokenizer, cache_size_map=True)
    inverted_index.build()

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)
    prog_bar = pyprind.ProgBar(len(rtable))

    for r_row in rtable_dict.values():
        r_id = r_row[r_key_attr_index]
        r_string = str(r_row[r_join_attr_index])

        r_join_attr_tokens = tokenize(r_string, tokenizer)
        r_num_tokens = len(r_join_attr_tokens)

        # probe inverted index and find overlap of candidates 
        candidate_overlap = _find_candidates(r_join_attr_tokens,
                                             inverted_index)

        for cand, overlap in iteritems(candidate_overlap):
            sim_score = (float(overlap) /
                         float(min(r_num_tokens,
                                   inverted_index.size_map[cand])))
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

    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,
                                                  l_out_attrs, r_out_attrs,
                                                  l_out_prefix, r_out_prefix)
    if out_sim_score:
        output_header.append("_sim_score")

    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table
