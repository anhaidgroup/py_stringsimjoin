# overlap coefficient join
from joblib import delayed, Parallel
from six import iteritems
import pandas as pd
import pyprind

from py_stringsimjoin.filter.overlap_filter import OverlapFilter
from py_stringsimjoin.index.inverted_index import InvertedIndex
from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_array, \
    find_output_attribute_indices, get_attrs_to_project, \
    get_num_processes_to_launch, get_output_header_from_tables, \
    get_output_row_from_tables, remove_redundant_attrs, split_table, COMP_OP_MAP
from py_stringsimjoin.utils.missing_value_handler import \
    get_pairs_with_missing_value
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_comp_op_for_sim_measure, validate_key_attr, \
    validate_input_table, validate_threshold, validate_tokenizer, \
    validate_output_attrs


def overlap_coefficient_join_py(ltable, rtable,
                                l_key_attr, r_key_attr,
                                l_join_attr, r_join_attr,
                                tokenizer, threshold, comp_op='>=',
                                allow_empty=True, allow_missing=False,
                                l_out_attrs=None, r_out_attrs=None,
                                l_out_prefix='l_', r_out_prefix='r_',
                                out_sim_score=True, n_jobs=1, show_progress=True):
    """Join two tables using overlap coefficient.

    For two sets X and Y, the overlap coefficient between them is given by:                      
                                                                                
        :math:`overlap\\_coefficient(X, Y) = \\frac{|X \\cap Y|}{\\min(|X|, |Y|)}`             

    In the case where one of X and Y is an empty set and the other is a 
    non-empty set, we define their overlap coefficient to be 0. In the case 
    where both X and Y are empty sets, we define their overlap coefficient to 
    be 1.                                                                                 

    Finds tuple pairs from left table and right table such that the overlap 
    coefficient between the join attributes satisfies the condition on input 
    threshold. For example, if the comparison operator is '>=', finds tuple     
    pairs whose overlap coefficient between the strings that are the values of    
    the join attributes is greater than or equal to the input threshold, as     
    specified in "threshold". 

    Args:
        ltable (DataFrame): left input table.

        rtable (DataFrame): right input table.

        l_key_attr (string): key attribute in left table.

        r_key_attr (string): key attribute in right table.

        l_join_attr (string): join attribute in left table.

        r_join_attr (string): join attribute in right table.

        tokenizer (Tokenizer): tokenizer to be used to tokenize join     
            attributes.                                                         
                                                                                
        threshold (float): overlap coefficient threshold to be satisfied.        
                                                                                
        comp_op (string): comparison operator. Supported values are '>=', '>'   
            and '=' (defaults to '>=').                                         
                                                                                
        allow_empty (boolean): flag to indicate whether tuple pairs with empty  
            set of tokens in both the join attributes should be included in the 
            output (defaults to True).                                          
                                                                                
        allow_missing (boolean): flag to indicate whether tuple pairs with      
            missing value in at least one of the join attributes should be      
            included in the output (defaults to False). If this flag is set to  
            True, a tuple in ltable with missing value in the join attribute    
            will be matched with every tuple in rtable and vice versa.          
                                                                                
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
        An output table containing tuple pairs that satisfy the join            
        condition (DataFrame).  
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

    # check if the join attributes are not of numeric type                      
    validate_attr_type(l_join_attr, ltable[l_join_attr].dtype,                  
                       'join attribute', 'left table')                          
    validate_attr_type(r_join_attr, rtable[r_join_attr].dtype,                  
                       'join attribute', 'right table')

    # check if the input tokenizer is valid
    validate_tokenizer(tokenizer)

    # check if the input threshold is valid
    validate_threshold(threshold, 'OVERLAP_COEFFICIENT')

    # check if the comparison operator is valid
    validate_comp_op_for_sim_measure(comp_op, 'OVERLAP_COEFFICIENT')

    # check if the output attributes exist
    validate_output_attrs(l_out_attrs, ltable.columns,
                          r_out_attrs, rtable.columns)

    # check if the key attributes are unique and do not contain missing values
    validate_key_attr(l_key_attr, ltable, 'left table')
    validate_key_attr(r_key_attr, rtable, 'right table')

    # set return_set flag of tokenizer to be True, in case it is set to False
    revert_tokenizer_return_set_flag = False
    if not tokenizer.get_return_set():
        tokenizer.set_return_set(True)
        revert_tokenizer_return_set_flag = True

    # remove redundant attrs from output attrs.
    l_out_attrs = remove_redundant_attrs(l_out_attrs, l_key_attr)
    r_out_attrs = remove_redundant_attrs(r_out_attrs, r_key_attr)

    # get attributes to project.  
    l_proj_attrs = get_attrs_to_project(l_out_attrs, l_key_attr, l_join_attr)
    r_proj_attrs = get_attrs_to_project(r_out_attrs, r_key_attr, r_join_attr)

    # Do a projection on the input dataframes to keep only the required         
    # attributes. Then, remove rows with missing value in join attribute from   
    # the input dataframes. Then, convert the resulting dataframes into ndarray.
    ltable_array = convert_dataframe_to_array(ltable, l_proj_attrs, l_join_attr)
    rtable_array = convert_dataframe_to_array(rtable, r_proj_attrs, r_join_attr)

    # computes the actual number of jobs to launch.
    n_jobs = min(get_num_processes_to_launch(n_jobs), len(rtable_array))

    if n_jobs <= 1:
        # if n_jobs is 1, do not use any parallel code.
        output_table = _overlap_coefficient_join_split(
                                           ltable_array, rtable_array,
                                           l_proj_attrs, r_proj_attrs,
                                           l_key_attr, r_key_attr,
                                           l_join_attr, r_join_attr,
                                           tokenizer, threshold, comp_op,
                                           allow_empty,
                                           l_out_attrs, r_out_attrs,
                                           l_out_prefix, r_out_prefix,
                                           out_sim_score, show_progress)
    else:
        # if n_jobs is above 1, split the right table into n_jobs splits and    
        # join each right table split with the whole of left table in a separate
        # process.
        r_splits = split_table(rtable_array, n_jobs)
        results = Parallel(n_jobs=n_jobs)(
                                delayed(_overlap_coefficient_join_split)(
                                    ltable_array, r_splits[job_index],
                                    l_proj_attrs, r_proj_attrs,
                                    l_key_attr, r_key_attr,
                                    l_join_attr, r_join_attr,
                                    tokenizer, threshold, comp_op,
                                    allow_empty,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score,
                                    (show_progress and (job_index==n_jobs-1)))
                                for job_index in range(n_jobs))
        output_table = pd.concat(results)

    # If allow_missing flag is set, then compute all pairs with missing value in
    # at least one of the join attributes and then add it to the output         
    # obtained from the join. 
    if allow_missing:
        missing_pairs = get_pairs_with_missing_value(
                                            ltable, rtable,
                                            l_key_attr, r_key_attr,
                                            l_join_attr, r_join_attr,
                                            l_out_attrs, r_out_attrs,
                                            l_out_prefix, r_out_prefix,
                                            out_sim_score, show_progress)
        output_table = pd.concat([output_table, missing_pairs])

    # add an id column named '_id' to the output table.
    output_table.insert(0, '_id', range(0, len(output_table)))

    # revert the return_set flag of tokenizer, in case it was modified.
    if revert_tokenizer_return_set_flag:
        tokenizer.set_return_set(False)

    return output_table


def _overlap_coefficient_join_split(ltable_list, rtable_list,
                                    l_columns, r_columns,
                                    l_key_attr, r_key_attr,
                                    l_join_attr, r_join_attr,
                                    tokenizer, threshold, comp_op,
                                    allow_empty,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score, show_progress):
    """Perform overlap coefficient join for a split of ltable and rtable"""
    # find column indices of key attr, join attr and output attrs in ltable
    l_key_attr_index = l_columns.index(l_key_attr)
    l_join_attr_index = l_columns.index(l_join_attr)
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, join attr and output attrs in rtable
    r_key_attr_index = r_columns.index(r_key_attr)
    r_join_attr_index = r_columns.index(r_join_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # Build inverted index over ltable
    inverted_index = InvertedIndex(ltable_list, l_join_attr_index,
                                   tokenizer, cache_size_flag=True)
    # While building the index, we cache the record ids with empty set of 
    # tokens. This is needed to handle the allow_empty flag.
    cached_data = inverted_index.build(allow_empty)
    l_empty_records = cached_data['empty_records']

    overlap_filter = OverlapFilter(tokenizer, 1)
    comp_fn = COMP_OP_MAP[comp_op]

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)

    if show_progress:
        prog_bar = pyprind.ProgBar(len(rtable_list))

    for r_row in rtable_list:
        r_string = r_row[r_join_attr_index]

        r_join_attr_tokens = tokenizer.tokenize(r_string)
        r_num_tokens = len(r_join_attr_tokens)

        # If allow_empty flag is set and the current rtable record has empty set
        # of tokens in the join attribute, then generate output pairs joining   
        # the current rtable record with those records in ltable with empty set 
        # of tokens in the join attribute. These ltable record ids are cached in
        # l_empty_records list which was constructed when building the inverted 
        # index.
        if allow_empty and r_num_tokens == 0:
            for l_id in l_empty_records:
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                                     ltable_list[l_id], r_row,
                                     l_key_attr_index, r_key_attr_index,
                                     l_out_attrs_indices,
                                     r_out_attrs_indices)
                else:
                    output_row = [ltable_list[l_id][l_key_attr_index],
                                  r_row[r_key_attr_index]]

                if out_sim_score:
                    output_row.append(1.0)
                output_rows.append(output_row)
            continue

        # probe inverted index and find overlap of candidates 
        candidate_overlap = overlap_filter.find_candidates(
                                r_join_attr_tokens, inverted_index)

        for cand, overlap in iteritems(candidate_overlap):
            # compute the actual similarity score                           
            sim_score = (float(overlap) /
                         float(min(r_num_tokens,
                                   inverted_index.size_cache[cand])))

            if comp_fn(sim_score, threshold):
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                                     ltable_list[cand], r_row,
                                     l_key_attr_index, r_key_attr_index,
                                     l_out_attrs_indices, r_out_attrs_indices)
                else:
                    output_row = [ltable_list[cand][l_key_attr_index],
                                  r_row[r_key_attr_index]]

                # if out_sim_score flag is set, append the overlap coefficient 
                # score to the output record.  
                if out_sim_score:
                    output_row.append(sim_score)

                output_rows.append(output_row)

        if show_progress:
            prog_bar.update()

    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,
                                                  l_out_attrs, r_out_attrs,
                                                  l_out_prefix, r_out_prefix)
    if out_sim_score:
        output_header.append("_sim_score")

    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table
