
import operator
import types

from joblib import delayed, Parallel
from six.moves import copyreg
import numpy as np
import pandas as pd
import pyprind

from py_stringsimjoin.utils.generic_helper import build_dict_from_table, \
    find_output_attribute_indices, get_attrs_to_project, \
    get_num_processes_to_launch, get_output_header_from_tables, \
    get_output_row_from_tables, remove_redundant_attrs, split_table, COMP_OP_MAP
from py_stringsimjoin.utils.pickle import pickle_instance_method, \
                                          unpickle_instance_method
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_comp_op, validate_key_attr, validate_input_table, \
    validate_tokenizer, validate_output_attrs


# Register pickle and unpickle methods for handling instance methods.
# This is because joblib doesn't pickle instance methods, by default.
# Hence, if the sim_function supplied to apply_matcher is an instance
# method, it will result in an error. To avoid this, we register custom
# functions to pickle and unpickle instance methods. 
copyreg.pickle(types.MethodType,
               pickle_instance_method,
               unpickle_instance_method)


def apply_matcher(candset,
                  candset_l_key_attr, candset_r_key_attr,
                  ltable, rtable,
                  l_key_attr, r_key_attr,
                  l_match_attr, r_match_attr,
                  tokenizer, sim_function,
                  threshold, comp_op='>=',
                  allow_missing=False,
                  l_out_attrs=None, r_out_attrs=None,
                  l_out_prefix='l_', r_out_prefix='r_',
                  out_sim_score=True, n_jobs=1, show_progress=True):
    """Find matching string pairs from the candidate set (typically produced by
    applying a filter to two tables) by applying a matcher of form 
    (sim_function comp_op threshold).

    Specifically, this method computes the input similarity function on string 
    pairs in the candidate set and checks if the resulting score satisfies the 
    input threshold (depending on the comparison operator).

    Args:
        candset (DataFrame): input candidate set.

        candset_l_key_attr (string): attribute in candidate set which is a key 
            in left table.

        candset_r_key_attr (string): attribute in candidate set which is a key 
            in right table.

        ltable (DataFrame): left input table.

        rtable (DataFrame): right input table.

        l_key_attr (string): key attribute in left table.

        r_key_attr (string): key attribute in right table.

        l_match_attr (string): attribute in left table on which the matcher 
            should be applied.

        r_match_attr (string): attribute in right table on which the matcher
            should be applied.

        tokenizer (Tokenizer): tokenizer to be used to tokenize the
            match attributes. If set to None, the matcher is applied directly
            on the match attributes.

        sim_function (function): matcher function to be applied.

        threshold (float): threshold to be satisfied.

        comp_op (string): comparison operator. Supported values are '>=', '>', '
            <=', '<', '=' and '!=' (defaults to '>=').

        allow_missing (boolean): flag to indicate whether tuple pairs with 
            missing value in at least one of the match attributes should be 
            included in the output (defaults to False). 

        l_out_attrs (list): list of attribute names from the left table to be 
            included in the output table (defaults to None).

        r_out_attrs (list): list of attribute names from the right table to be 
            included in the output table (defaults to None).

        l_out_prefix (string): prefix to be used for the attribute names coming 
            from the left table, in the output table (defaults to 'l\_').

        r_out_prefix (string): prefix to be used for the attribute names coming 
            from the right table, in the output table (defaults to 'r\_').

        out_sim_score (boolean): flag to indicate whether similarity score 
            should be included in the output table (defaults to True). Setting
            this flag to True will add a column named '_sim_score' in the 
            output table. This column will contain the similarity scores for the
            tuple pairs in the output. 

        n_jobs (int): number of parallel jobs to use for the computation        
            (defaults to 1). If -1 is given, all CPUs are used. If 1 is given,  
            no parallel computing code is used at all, which is useful for      
            debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used      
            (where n_cpus is the total number of CPUs in the machine). Thus for 
            n_jobs = -2, all CPUs but one are used. If (n_cpus + 1 + n_jobs)    
            becomes less than 1, then no parallel computing code will be used   
            (i.e., equivalent to the default). 

        show_progress (boolean): flag to indicate whether task progress should 
            be displayed to the user (defaults to True).

    Returns:
        An output table containing tuple pairs from the candidate set that 
        survive the matcher (DataFrame).
    """

    # check if the input candset is a dataframe
    validate_input_table(candset, 'candset')

    # check if the candset key attributes exist
    validate_attr(candset_l_key_attr, candset.columns,
                  'left key attribute', 'candset')
    validate_attr(candset_r_key_attr, candset.columns,
                  'right key attribute', 'candset')

    # check if the input tables are dataframes
    validate_input_table(ltable, 'left table')
    validate_input_table(rtable, 'right table')

    # check if the key attributes and join attributes exist
    validate_attr(l_key_attr, ltable.columns,
                  'key attribute', 'left table')
    validate_attr(r_key_attr, rtable.columns,
                  'key attribute', 'right table')
    validate_attr(l_match_attr, ltable.columns,
                  'match attribute', 'left table')
    validate_attr(r_match_attr, rtable.columns,
                  'match attribute', 'right table')

    # check if the output attributes exist
    validate_output_attrs(l_out_attrs, ltable.columns,
                          r_out_attrs, rtable.columns)

    # check if the input tokenizer is valid, if it is not None
    if tokenizer is not None:
        validate_tokenizer(tokenizer)

    # check if the comparison operator is valid
    validate_comp_op(comp_op)

    # check if the key attributes are unique and do not contain missing values
    validate_key_attr(l_key_attr, ltable, 'left table')
    validate_key_attr(r_key_attr, rtable, 'right table')

    # check for empty candset
    if candset.empty:
        return candset

    # remove redundant attrs from output attrs.
    l_out_attrs = remove_redundant_attrs(l_out_attrs, l_key_attr)
    r_out_attrs = remove_redundant_attrs(r_out_attrs, r_key_attr)

    # get attributes to project.  
    l_proj_attrs = get_attrs_to_project(l_out_attrs, l_key_attr, l_match_attr)
    r_proj_attrs = get_attrs_to_project(r_out_attrs, r_key_attr, r_match_attr)

    # do a projection on the input dataframes. Note that this doesn't create a 
    # copy of the dataframes. It only creates a view on original dataframes.
    ltable_projected = ltable[l_proj_attrs]
    rtable_projected = rtable[r_proj_attrs]

    # computes the actual number of jobs to launch.
    n_jobs = min(get_num_processes_to_launch(n_jobs), len(candset))

    # If a tokenizer is provided, we can optimize by tokenizing each value 
    # only once by caching the tokens of l_match_attr and r_match_attr. But, 
    # this can be a bad strategy in case the candset has very few records 
    # compared to the original tables. Hence, we check if the sum of tuples in 
    # ltable and rtable is less than twice the number of tuples in the candset. 
    # If yes, we decide to cache the token values. Else, we do not cache the 
    # tokens as the candset is small.
    l_tokens = None
    r_tokens = None
    if tokenizer is not None and (len(ltable) + len(rtable) < len(candset)*2):
        l_tokens = generate_tokens(ltable_projected, l_key_attr, l_match_attr,
                                   tokenizer)
        r_tokens = generate_tokens(rtable_projected, r_key_attr, r_match_attr,
                                   tokenizer)

    if n_jobs <= 1:
        # if n_jobs is 1, do not use any parallel code.                     
        output_table =  _apply_matcher_split(candset,
                                    candset_l_key_attr, candset_r_key_attr,
                                    ltable_projected, rtable_projected,
                                    l_key_attr, r_key_attr,
                                    l_match_attr, r_match_attr,
                                    tokenizer, sim_function,
                                    threshold, comp_op, allow_missing,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score, show_progress,
                                    l_tokens, r_tokens)
    else:
        # if n_jobs is above 1, split the candset into n_jobs splits and apply   
        # the matcher on each candset split in a separate process.  
        candset_splits = split_table(candset, n_jobs)
        results = Parallel(n_jobs=n_jobs)(delayed(_apply_matcher_split)(
                                      candset_splits[job_index],
                                      candset_l_key_attr, candset_r_key_attr,
                                      ltable_projected, rtable_projected,
                                      l_key_attr, r_key_attr,
                                      l_match_attr, r_match_attr,
                                      tokenizer, sim_function,
                                      threshold, comp_op, allow_missing,
                                      l_out_attrs, r_out_attrs,
                                      l_out_prefix, r_out_prefix,
                                      out_sim_score,
                                      (show_progress and (job_index==n_jobs-1)),
                                      l_tokens, r_tokens)
                                          for job_index in range(n_jobs))
        output_table =  pd.concat(results)

    return output_table


def _apply_matcher_split(candset,
                         candset_l_key_attr, candset_r_key_attr,
                         ltable, rtable,
                         l_key_attr, r_key_attr,
                         l_match_attr, r_match_attr,
                         tokenizer, sim_function,
                         threshold, comp_op, allow_missing,
                         l_out_attrs, r_out_attrs,
                         l_out_prefix, r_out_prefix,
                         out_sim_score, show_progress, l_tokens, r_tokens):
    # find column indices of key attr, join attr and output attrs in ltable
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    l_match_attr_index = l_columns.index(l_match_attr)
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, join attr and output attrs in rtable
    r_columns = list(rtable.columns.values)
    r_key_attr_index = r_columns.index(r_key_attr)
    r_match_attr_index = r_columns.index(r_match_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # Build a dictionary on ltable
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index,
                                        l_match_attr_index, remove_null=False)

    # Build a dictionary on rtable
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index,
                                        r_match_attr_index, remove_null=False)

    # Find indices of l_key_attr and r_key_attr in candset
    candset_columns = list(candset.columns.values)
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)

    comp_fn = COMP_OP_MAP[comp_op]
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None) 

    output_rows = []

    if show_progress:
        prog_bar = pyprind.ProgBar(len(candset))

    tokenize_flag = False
    if tokenizer is not None:
        tokenize_flag =  True
        use_cache = False
        # check if we have cached the tokens.
        if l_tokens is not None and r_tokens is not None:
            use_cache = True

    for candset_row in candset.itertuples(index = False):
        l_id = candset_row[candset_l_key_attr_index]
        r_id = candset_row[candset_r_key_attr_index]

        l_row = ltable_dict[l_id]
        r_row = rtable_dict[r_id]
        
        l_apply_col_value = l_row[l_match_attr_index]
        r_apply_col_value = r_row[r_match_attr_index]

        allow_pair = False
        # Check if one of the inputs is missing. If yes, check the allow_missing
        # flag. If it is True, then add the pair to output. Else, continue.
        # If none of the input is missing, then proceed to apply the 
        # sim_function. 
        if pd.isnull(l_apply_col_value) or pd.isnull(r_apply_col_value):
            if allow_missing:
                allow_pair = True
                sim_score = np.NaN
            else:
                continue   
        else:
            if tokenize_flag:
                # If we have cached the tokens, we use it directly. Else, we
                # tokenize the values.
                if use_cache:
                    l_apply_col_value = l_tokens[l_id]
                    r_apply_col_value = r_tokens[r_id]
                else:
                    l_apply_col_value = tokenizer.tokenize(l_apply_col_value)
                    r_apply_col_value = tokenizer.tokenize(r_apply_col_value)
        
            sim_score = sim_function(l_apply_col_value, r_apply_col_value)
            allow_pair = comp_fn(sim_score, threshold)

        if allow_pair: 
            if has_output_attributes:
                output_row = get_output_row_from_tables(
                                 l_row, r_row,
                                 l_key_attr_index, r_key_attr_index,
                                 l_out_attrs_indices,
                                 r_out_attrs_indices)
                output_row.insert(0, candset_row[0])
            else:
                output_row = [candset_row[0], l_id, r_id]
            if out_sim_score:
                output_row.append(sim_score)
            output_rows.append(output_row)

        if show_progress:                    
            prog_bar.update()

    output_header = get_output_header_from_tables(
                        l_key_attr, r_key_attr,
                        l_out_attrs, r_out_attrs,
                        l_out_prefix, r_out_prefix)
    output_header.insert(0, '_id')
    if out_sim_score:
        output_header.append("_sim_score")

    # generate a dataframe from the list of output rows
    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table


def generate_tokens(table, key_attr, join_attr, tokenizer):
    table_nonnull = table[pd.notnull(table[join_attr])]
    return dict(zip(table_nonnull[key_attr],
                    table_nonnull[join_attr].apply(tokenizer.tokenize)))
