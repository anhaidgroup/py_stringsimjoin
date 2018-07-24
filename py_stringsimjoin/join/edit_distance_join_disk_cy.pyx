# edit distance join
# cython: linetrace = True
# distutils: define_macros = CYTHON_TRACE_NOGIL = 1

import math
import os
import pyprind
import shutil
import datetime
import pandas as pd
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from joblib import delayed, Parallel

from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_array, \
    find_output_attribute_indices, get_attrs_to_project, \
    get_num_processes_to_launch, get_output_header_from_tables, \
    get_output_row_from_tables, remove_non_ascii, remove_redundant_attrs, \
    split_table
from py_stringsimjoin.utils.missing_value_handler_disk import \
    get_pairs_with_missing_value_disk
from py_stringsimjoin.utils.token_ordering import \
    gen_token_ordering_for_tables, order_using_token_ordering
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_comp_op_for_sim_measure, validate_key_attr, \
    validate_input_table, validate_threshold, \
    validate_tokenizer_for_sim_measure, validate_output_attrs,validate_path,validate_output_file_path, \
    validate_data_limit

# Cython imports
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     

from py_stringsimjoin.similarity_measure.edit_distance cimport edit_distance
from py_stringsimjoin.index.inverted_index_cy cimport InvertedIndexCy             
from py_stringsimjoin.utils.cython_utils cimport compfnptr,\
    get_comparison_function, get_comp_type, int_min

def edit_distance_join_disk_cy(ltable, rtable,
                          l_key_attr, r_key_attr,
                          l_join_attr, r_join_attr,
                          double threshold,data_limit,
                          comp_op, allow_missing,
                          l_out_attrs, r_out_attrs,
                          l_out_prefix, r_out_prefix,
                          out_sim_score, int n_jobs,
                          bool show_progress, tokenizer,
                          temp_dir, output_file_path):

    """Join two tables using edit distance measure.

    This is the disk version of the previous edit_distance_join api.
    There can be a scenario that while performing join on large datasets,
    the intermediate in-memory data structures grow very large and thus lead
    to termination of the program due to insufficient memory. Keeping this problem
    in mind, edit_distance_join_disk is the updated version of the older
    edit_distance_join function that solves the above mentioned problem.
    So if the analysis is being done on the machine with small memory limits or
    if the input tables are too large, then this new edit_distance_join_disk can be
    used to avoid memory exceeding problem while processing.


    It Finds tuple pairs from left table and right table such that the edit
    distance between the join attributes satisfies the condition on input 
    threshold. For example, if the comparison operator is '<=', finds tuple     
    pairs whose edit distance between the strings that are the values of    
    the join attributes is less than or equal to the input threshold, as     
    specified in "threshold". 

    Note:
        Currently, this method only computes an approximate join result. This is
        because, to perform the join we transform an edit distance measure 
        between strings into an overlap measure between qgrams of the strings. 
        Hence, we need at least one qgram to be in common between two input 
        strings, to appear in the join output. For smaller strings, where all 
        qgrams of the strings differ, we cannot process them.
 
        This method implements a simplified version of the algorithm proposed in
        `Ed-Join: An Efficient Algorithm for Similarity Joins With Edit Distance
        Constraints (Chuan Xiao, Wei Wang and Xuemin Lin), VLDB 08
        <http://www.vldb.org/pvldb/1/1453957.pdf>`_. 
        
    Args:
        ltable (DataFrame): left input table.

        rtable (DataFrame): right input table.

        l_key_attr (string): key attribute in left table.

        r_key_attr (string): key attribute in right table.

        l_join_attr (string): join attribute in left table.

        r_join_attr (string): join attribute in right table.

        threshold (float): edit distance threshold to be satisfied.

        data_limit (int): threshold value for number of rows that would be kept
            in memory before writing the output on the disk. This is the maximum sum
            total of all rows that can be present in memory across all processes at
            a time. This is a new argument compared to edit distance join.
            (defaults to 1M)

        comp_op (string): comparison operator. Supported values are '<=', '<'   
            and '=' (defaults to '<=').                                         
                                                                                
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
                                                                                
        out_sim_score (boolean): flag to indicate whether the edit distance 
            score should be included in the output table (defaults to True). 
            Setting this flag to True will add a column named '_sim_score' in 
            the output table. This column will contain the edit distance scores 
            for the tuple pairs in the output.                                          

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

        tokenizer (Tokenizer): tokenizer to be used to tokenize the join 
            attributes during filtering, when edit distance measure is          
            transformed into an overlap measure. This must be a q-gram tokenizer
            (defaults to 2-gram tokenizer).

        temp_dir (string): absolute path where all the intermediate files will be generated.
            This is a new argument compared to edit distance join. (defaults to the current
            working directory).

        output_file_path (string): absolute path where the output file will be generated.
            Older file with same path and name will be removed. This is a new argument compared
            to edit distance join. (defaults to the current working directory/$default_output_file_name).

    Returns:                                                                    
        Returns the status of the computation. True if successfully completed else False (boolean).
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

    # check if the input tokenizer is valid for edit distance measure. Only
    # qgram tokenizer can be used for edit distance.
    validate_tokenizer_for_sim_measure(tokenizer, 'EDIT_DISTANCE')

    # check if the input threshold is valid
    validate_threshold(threshold, 'EDIT_DISTANCE')

    #check if the given datalimit is valid
    validate_data_limit(data_limit)

    # check if the comparison operator is valid
    validate_comp_op_for_sim_measure(comp_op, 'EDIT_DISTANCE')

    # check if the output attributes exist
    validate_output_attrs(l_out_attrs, ltable.columns,
                          r_out_attrs, rtable.columns)

    # check if the key attributes are unique and do not contain missing values
    validate_key_attr(l_key_attr, ltable, 'left table')
    validate_key_attr(r_key_attr, rtable, 'right table')

    #Check if the given path is valid
    validate_path(temp_dir)

    #Check if the given output file path is valid
    validate_output_file_path(output_file_path)

    # convert threshold to integer (incase if it is float)
    threshold = int(math.floor(threshold))

    # set return_set flag of tokenizer to be False, in case it is set to True
    revert_tokenizer_return_set_flag = False
    if tokenizer.get_return_set():
        tokenizer.set_return_set(False)
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
    cdef int index_count = 0
    cdef int iter
    data_limit_per_core = math.floor(data_limit/n_jobs)
    time_string = datetime.datetime.now().strftime("%H:%M:%S:%f")
    file_names = [str(i) + "_" + time_string + ".csv" for i in range(n_jobs)]
    file_names = [os.path.join(temp_dir,fname) for fname in file_names]


    if n_jobs <= 1:                                                             
        # if n_jobs is 1, do not use any parallel code.                         
        result = _edit_distance_join_split(                               
                               ltable_array, rtable_array,                      
                               l_proj_attrs, r_proj_attrs,                      
                               l_key_attr, r_key_attr,                          
                               l_join_attr, r_join_attr,                        
                               tokenizer, threshold,
                               comp_op, l_out_attrs,
                               r_out_attrs, l_out_prefix,
                               r_out_prefix, out_sim_score,
                               show_progress, 0,
                               temp_dir, data_limit_per_core,
                               file_names)
        results = []
        results.append(result)
    else:                                                                       
        # if n_jobs is above 1, split the right table into n_jobs splits and    
        # join each right table split with the whole of left table in a separate
        # process.                                                              
        r_splits = split_table(rtable_array, n_jobs)                            
        results = Parallel(n_jobs=n_jobs)(delayed(_edit_distance_join_split)(   
                                    ltable_array, r_splits[job_index],          
                                    l_proj_attrs, r_proj_attrs,                 
                                    l_key_attr, r_key_attr,                     
                                    l_join_attr, r_join_attr,                   
                                    tokenizer, threshold,
                                    comp_op, l_out_attrs,
                                    r_out_attrs, l_out_prefix,
                                    r_out_prefix, out_sim_score,
                                    (show_progress and (job_index==n_jobs-1)), job_index,
                                    temp_dir, data_limit_per_core,
                                    file_names)
                                for job_index in range(n_jobs)) 
    
    # If one of the parallel joins fail, clean up and return.
    if False in results:
        for fname in file_names:
            os.remove(fname)
        return False

    # Delete the file with the same name as output file, if it exists.
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)

    output_header = get_output_header_from_tables(
                       l_key_attr, r_key_attr,
                       l_out_attrs, r_out_attrs,
                       l_out_prefix, r_out_prefix)
    if out_sim_score:
        output_header.append("_sim_score")

    # Combine all the files from results into a single output file 
    # and remove those temporary files
    with open(output_file_path,'w+') as outfile :
        outfile.write((",".join(output_header)))
        outfile.write("\n")
        try:
            for fname in file_names:
                with open(fname,'r') as infile :
                    shutil.copyfileobj(infile,outfile)
                os.remove(fname)
        except Exception as e:        
            # removing all the intermediate files before returning
            for fname in file_names:
                if os.path.isfile(fname):
                    os.remove(fname)
            # removing output file if it exists
            if os.path.isfile(output_file_path):
                os.remove(output_file_path)
            return False




    # If allow_missing flag is set, then compute all pairs with missing value in
    # at least one of the join attributes and then add it to the output         
    # obtained from the join.                                                   
    if allow_missing:
        missing_pairs_file_name = "missing_pairs.csv" + datetime.datetime.now().strftime("%H:%M:%S.%f" + ".csv")
        missing_pairs_file_name = os.path.join(temp_dir, missing_pairs_file_name)
        if os.path.isfile(missing_pairs_file_name):
            os.remove(missing_pairs_file_name)

        missing_status = get_pairs_with_missing_value_disk(
                                             ltable, rtable,
                                           l_key_attr, r_key_attr,
                                           l_join_attr, r_join_attr,
                                           temp_dir, data_limit_per_core,
                                           missing_pairs_file_name, l_out_attrs,
                                           r_out_attrs, l_out_prefix,
                                           r_out_prefix, out_sim_score,
                                           show_progress,)

        # Write missing pairs to the output file
        with open(output_file_path,'a+') as outfile :
            try:
                with open(missing_pairs_file_name,'r') as infile :
                    shutil.copyfileobj(infile,outfile)
                os.remove(missing_pairs_file_name)
            except Exception as e:
                print(str(e))

                # removing all the intermediate files before returning
                if os.path.isfile(missing_pairs_file_name):
                    os.remove(missing_pairs_file_name)
                # removing output file if it exists
                if os.path.isfile(output_file_path):
                    os.remove(output_file_path)
                return False

    # revert the return_set flag of tokenizer, in case it was modified.         
    if revert_tokenizer_return_set_flag:                                        
        tokenizer.set_return_set(True)                                          
                                                                                
    return True


def _edit_distance_join_split(ltable_array, rtable_array,                         
                              l_columns, r_columns,                             
                              l_key_attr, r_key_attr,                           
                              l_join_attr, r_join_attr,                         
                              tokenizer, threshold,
                              comp_op, l_out_attrs,
                              r_out_attrs, l_out_prefix,
                              r_out_prefix, out_sim_score,
                              show_progress, job_index,
                              dir, data_limit_per_core,
                              file_names):
    """Perform edit distance join for a split of ltable and rtable"""
  
    # find column indices of key attr, join attr and output attrs in ltable
    l_key_attr_index = l_columns.index(l_key_attr)
    l_join_attr_index = l_columns.index(l_join_attr)
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, join attr and output attrs in rtable
    r_key_attr_index = r_columns.index(r_key_attr)
    r_join_attr_index = r_columns.index(r_join_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    sim_measure_type = 'EDIT_DISTANCE'
    # generate token ordering using tokens in l_join_attr
    # and r_join_attr
    token_ordering = gen_token_ordering_for_tables(
                         [ltable_array, rtable_array],
                         [l_join_attr_index, r_join_attr_index],
                         tokenizer, sim_measure_type)

    cdef vector[string] lstrings
    cdef InvertedIndexCy prefix_index = InvertedIndexCy()                                           

    str2bytes = lambda x: x if isinstance(x, bytes) else x.encode('utf-8')

    tokenize_and_build_index(ltable_array, l_join_attr_index,             
                             tokenizer, threshold, str2bytes, token_ordering,            
                             lstrings, prefix_index)

    cdef vector[vector[int]] rtokens
    cdef vector[string] rstrings
    output_temp_file = open(file_names[job_index],'a+')

    for r_row in rtable_array:
        rstring = r_row[r_join_attr_index]
        rstrings.push_back(str2bytes(rstring))
        # tokenize string and order the tokens using the token ordering
        rstring_tokens = order_using_token_ordering(
            tokenizer.tokenize(rstring), token_ordering)
        rtokens.push_back(rstring_tokens)

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)

    if show_progress:                                                           
        prog_bar = pyprind.ProgBar(len(rtable_array))   

    cdef oset[int] candidates                                                   
    cdef vector[int] tokens                                                     
    cdef int j, m, i, prefix_length, cand                                     
    cdef double edit_dist                                                       
    cdef int qval = tokenizer.qval                                              
    cdef compfnptr comp_fn
    comp_fn = get_comparison_function(get_comp_type(comp_op))

                                                                                
    try:
        for i in range(rtokens.size()):                          
            tokens = rtokens[i]                                                     
            m = tokens.size()                                                       
            prefix_length = int_min(<int>(qval * threshold + 1), m)                 
                                                                                    
            for j in range(prefix_length):                                          
                if prefix_index.index.find(tokens[j]) == prefix_index.index.end():                            
                    continue                                                        
                for cand in prefix_index.index[tokens[j]]:                                       
                    candidates.insert(cand)                                         
                                                                                    
            for cand in candidates:                                                 
                if m - threshold <= prefix_index.size_vector[cand] <= m + threshold:             
                    edit_dist = edit_distance(lstrings[cand], rstrings[i])          
                    if comp_fn(edit_dist, threshold):
                        if has_output_attributes:                                           
                            record = get_output_row_from_tables(
                                             ltable_array[cand], rtable_array[i],                                  
                                             l_key_attr_index, r_key_attr_index,            
                                             l_out_attrs_indices,                           
                                             r_out_attrs_indices)                           
                        else:                                                               
                            record = [ltable_array[cand][l_key_attr_index],
                                          rtable_array[i][r_key_attr_index]]                          
                                                                                    
                        # if out_sim_score flag is set, append the edit distance            
                        # score to the output record.                                       
                        if out_sim_score:                                                   
                            record.append(edit_dist)
                        output_rows.append(record)
    
                        #if the output rows id bigger than the given data limit, write to the file.
                        if len(output_rows)> data_limit_per_core :
                            df = pd.DataFrame(output_rows)
                            df.to_csv(output_temp_file, header = False, index = False)
                            output_rows = []
            candidates = []
    
            if show_progress:                                                       
                prog_bar.update()  
        # Write the remaining output rows left to the file.
        if len(output_rows) > 0 :
            df = pd.DataFrame(output_rows)
            df.to_csv(output_temp_file, header = False, index= False)
            output_rows = []
    except:
        output_temp_file.close()
        return False

    output_temp_file.close()
    return True


cdef void tokenize_and_build_index(ltable_array, l_join_attr_index,
                                   tokenizer, threshold, str2bytes, 
                                   token_ordering,
                                   vector[string]& lstrings,
                                   InvertedIndexCy index):
    cdef vector[vector[int]] ltokens
    for l_row in ltable_array:                                                  
        lstring = l_row[l_join_attr_index]                                      
        lstrings.push_back(str2bytes(lstring))                                  
        # tokenize string and order the tokens using the token ordering         
        lstring_tokens = order_using_token_ordering(                            
            tokenizer.tokenize(lstring), token_ordering)                        
        ltokens.push_back(lstring_tokens)

    index.build_prefix_index(ltokens, tokenizer.qval, threshold)         
