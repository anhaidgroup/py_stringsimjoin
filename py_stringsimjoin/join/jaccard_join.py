# jaccard join
import math
import tempfile
import csv
import pandas as pd
from joblib import delayed, Parallel

from py_stringsimjoin.join.set_sim_join import set_sim_join
from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_array, \
    get_attrs_to_project, get_num_processes_to_launch, remove_redundant_attrs, \
    split_table, get_temp_filenames, merge_outputs_and_add_id, add_id_to_file, remove_files
from py_stringsimjoin.utils.missing_value_handler import \
    get_pairs_with_missing_value
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_comp_op_for_sim_measure, validate_key_attr, \
    validate_input_table, validate_threshold, validate_tokenizer, \
    validate_output_attrs


def jaccard_join(ltable, rtable,
                 l_key_attr, r_key_attr,
                 l_join_attr, r_join_attr,
                 tokenizer, threshold, comp_op='>=',
                 allow_empty=True, allow_missing=False,
                 l_out_attrs=None, r_out_attrs=None,
                 l_out_prefix='l_', r_out_prefix='r_',
                 out_sim_score=True, n_jobs=1, show_progress=True,
                 file_name=None, mem_threshold=1e9, append_to_file=False
                 ):
    """Join two tables using Jaccard similarity measure.

    For two sets X and Y, the Jaccard similarity score between them is given by:                      
                                                                                
        :math:`jaccard(X, Y) = \\frac{|X \\cap Y|}{|X \\cup Y|}` 

    In the case where both X and Y are empty sets, we define their Jaccard 
    score to be 1. 

    Finds tuple pairs from left table and right table such that the Jaccard 
    similarity between the join attributes satisfies the condition on input 
    threshold. For example, if the comparison operator is '>=', finds tuple     
    pairs whose Jaccard similarity between the strings that are the values of    
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

        threshold (float): Jaccard similarity threshold to be satisfied.

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
        
        file_name (string): Location where output will be stored.
        
        mem_threshold (int): Memory Threshold which is the limit of
        intermediate data that can be written to memory.
        
        append_to_file (boolean): A flag to indicate whether appending
        is enabled or disabled.

    Returns:
        A boolean value to indicate completion of operation.
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
    validate_threshold(threshold, 'JACCARD')

    # check if the comparison operator is valid
    validate_comp_op_for_sim_measure(comp_op, 'JACCARD')

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
    
    # Call the respective join function depending on whether join is to
    # be done in-memory or out of memory
    output = None
    if file_name == None:
        #join in memory. Tuple pair functionality will not be used in this case.
        output = _jaccard_join_in_mem(ltable, rtable,
                                      ltable_array, rtable_array,
                                      l_proj_attrs, r_proj_attrs,
                                      l_key_attr, r_key_attr,
                                      l_join_attr, r_join_attr,
                                      tokenizer, threshold, comp_op,
                                      allow_empty, allow_missing,
                                      l_out_attrs, r_out_attrs,
                                      l_out_prefix, r_out_prefix,
                                      out_sim_score,
                                      n_jobs,
                                      show_progress,
                                      revert_tokenizer_return_set_flag)
    else:
        #join out of memory and using Tuple pair functionality.
        output = _jaccard_join_ooc_mem(ltable, rtable,
                                       ltable_array, rtable_array,
                                       l_proj_attrs, r_proj_attrs,
                                       l_key_attr, r_key_attr,
                                       l_join_attr, r_join_attr,
                                       tokenizer, threshold, comp_op,
                                       allow_empty, allow_missing,
                                       l_out_attrs, r_out_attrs,
                                       l_out_prefix, r_out_prefix,
                                       out_sim_score,
                                       n_jobs,
                                       show_progress,
                                       revert_tokenizer_return_set_flag,
                                       file_name, mem_threshold,
                                       append_to_file
                                       )

    return output

#Args are same as used in above functions
def _jaccard_join_ooc_mem(ltable, rtable,
                          ltable_array, rtable_array,
                          l_columns, r_columns,
                          l_key_attr, r_key_attr,
                          l_join_attr, r_join_attr,
                          tokenizer, threshold, comp_op,
                          allow_empty, allow_missing,
                          l_out_attrs, r_out_attrs,
                          l_out_prefix, r_out_prefix,
                          out_sim_score, n_jobs,
                          show_progress,
                          revert_tokenizer_return_set_flag,
                          file_name, mem_threshold, append_to_file
                          ):
    # computes the actual number of jobs to launch.
    n_jobs = min(get_num_processes_to_launch(n_jobs), len(rtable_array))
    if n_jobs <= 1:
        # if n_jobs is 1, do not use any parallel code.
        scratch_dir = tempfile.mkdtemp()
        # Intermediate file locations to buffer the intermediate parallel outputs.
        _intermediate_file_names = get_temp_filenames(1, scratch_dir)
        _intermediate_file_name = _intermediate_file_names[0]

        output = set_sim_join(ltable_array, rtable_array,
                              l_columns, r_columns,
                              l_key_attr, r_key_attr,
                              l_join_attr, r_join_attr,
                              tokenizer, 'JACCARD',
                              threshold, comp_op, allow_empty,
                              l_out_attrs, r_out_attrs,
                              l_out_prefix, r_out_prefix,
                              out_sim_score, show_progress,
                              _intermediate_file_name, mem_threshold, append_to_file)
        if allow_missing:
            missing_pairs = get_pairs_with_missing_value(
                ltable, rtable,
                l_key_attr, r_key_attr,
                l_join_attr, r_join_attr,
                l_out_attrs, r_out_attrs,
                l_out_prefix, r_out_prefix,
                out_sim_score, show_progress,
                _intermediate_file_name, mem_threshold, append_to_file=True)

        add_id_to_file(_intermediate_file_name, file_name, mem_threshold=mem_threshold)
        remove_files([_intermediate_file_name])



    else:
        # if n_jobs is above 1, split the right table into n_jobs splits and
        # join each right table split with the whole of left table in a separate
        # process.
        r_splits = split_table(rtable_array, n_jobs)
        scratch_dir = tempfile.mkdtemp()
        _intermediate_file_names = get_temp_filenames(n_jobs, scratch_dir)

        results = Parallel(n_jobs=n_jobs)(delayed(set_sim_join)(
            ltable_array, r_splits[job_index],
            l_columns, r_columns,
            l_key_attr, r_key_attr,
            l_join_attr, r_join_attr,
            tokenizer, 'JACCARD',
            threshold, comp_op, allow_empty,
            l_out_attrs, r_out_attrs,
            l_out_prefix, r_out_prefix,
            out_sim_score,
            (show_progress and (job_index == n_jobs - 1)),
            _intermediate_file_names[job_index],
            math.ceil(mem_threshold / 4),
            append_to_file)
            for job_index in range(n_jobs))

        if allow_missing:
            miss_file_names = get_temp_filenames(1, scratch_dir)
            miss_file_name = miss_file_names[0]
            get_pairs_with_missing_value(
                ltable, rtable,
                l_key_attr, r_key_attr,
                l_join_attr, r_join_attr,
                l_out_attrs, r_out_attrs,
                l_out_prefix, r_out_prefix,
                out_sim_score, show_progress,
                miss_file_name, mem_threshold, append_to_file=True)

            _intermediate_file_names.append(miss_file_name)


        status = merge_outputs_and_add_id(_intermediate_file_names, file_name,
                         mem_threshold=mem_threshold)

        remove_files(_intermediate_file_names)

    if revert_tokenizer_return_set_flag:
        tokenizer.set_return_set(False)

    # Do not return the dataframes here as size can be massive. We just need to return boolean value back to caller.
    return True

def _jaccard_join_in_mem(ltable, rtable,
                         ltable_array, rtable_array,
                         l_columns, r_columns,
                         l_key_attr, r_key_attr,
                         l_join_attr, r_join_attr,
                         tokenizer, threshold, comp_op,
                         allow_empty, allow_missing,
                         l_out_attrs, r_out_attrs,
                         l_out_prefix, r_out_prefix,
                         out_sim_score, n_jobs,
                         show_progress,
                         revert_tokenizer_return_set_flag):
    # computes the actual number of jobs to launch.
    n_jobs = min(get_num_processes_to_launch(n_jobs), len(rtable_array))
    if n_jobs <= 1:
        # if n_jobs is 1, do not use any parallel code.
        output_table = set_sim_join(ltable_array, rtable_array,
                                    l_columns, r_columns,
                                    l_key_attr, r_key_attr,
                                    l_join_attr, r_join_attr,
                                    tokenizer, 'JACCARD',
                                    threshold, comp_op, allow_empty,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score, show_progress)
    else:
        # if n_jobs is above 1, split the right table into n_jobs splits and
        # join each right table split with the whole of left table in a separate
        # process.
        r_splits = split_table(rtable_array, n_jobs)
        results = Parallel(n_jobs=n_jobs)(delayed(set_sim_join)(
            ltable_array, r_splits[job_index],
            l_columns, r_columns,
            l_key_attr, r_key_attr,
            l_join_attr, r_join_attr,
            tokenizer, 'JACCARD',
            threshold, comp_op, allow_empty,
            l_out_attrs, r_out_attrs,
            l_out_prefix, r_out_prefix,
            out_sim_score,
            (show_progress and (job_index == n_jobs - 1)))
                                          for job_index in range(n_jobs))
        output_table = pd.concat(results)
        # add an id column named '_id' to the output table.

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
    output_table.reset_index(drop=True, inplace=True)
    # revert the return_set flag of tokenizer, in case it was modified.
    if revert_tokenizer_return_set_flag:
        tokenizer.set_return_set(False)

    return output_table
