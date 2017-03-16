from py_stringsimjoin.filter.overlap_filter import OverlapFilter
from py_stringsimjoin.utils.validation import validate_tokenizer


def overlap_join_py(ltable, rtable,
                    l_key_attr, r_key_attr,
                    l_join_attr, r_join_attr,
                    tokenizer, threshold, comp_op='>=',
                    allow_missing=False,
                    l_out_attrs=None, r_out_attrs=None,
                    l_out_prefix='l_', r_out_prefix='r_',
                    out_sim_score=True, n_jobs=1, show_progress=True):
    """Join two tables using overlap measure.

    For two sets X and Y, the overlap between them is given by:                       
                                                                                
        :math:`overlap(X, Y) = |X \\cap Y|`

    Finds tuple pairs from left table and right table such that the overlap 
    between the join attributes satisfies the condition on input threshold. For 
    example, if the comparison operator is '>=', finds tuple pairs whose 
    overlap between the strings that are the values of the join attributes is 
    greater than or equal to the input threshold, as specified in "threshold".

    Args:
        ltable (DataFrame): left input table.

        rtable (DataFrame): right input table.

        l_key_attr (string): key attribute in left table.

        r_key_attr (string): key attribute in right table.

        l_join_attr (string): join attribute in left table.

        r_join_attr (string): join attribute in right table.

        tokenizer (Tokenizer): tokenizer to be used to tokenize join     
            attributes.                                                         
                                                                                
        threshold (float): overlap threshold to be satisfied.        
                                                                                
        comp_op (string): comparison operator. Supported values are '>=', '>'   
            and '=' (defaults to '>=').                                         
                                                                                
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

    # check if the input tokenizer is valid
    validate_tokenizer(tokenizer)

    # set return_set flag of tokenizer to be True, in case it is set to False
    revert_tokenizer_return_set_flag = False
    if not tokenizer.get_return_set():
        tokenizer.set_return_set(True)
        revert_tokenizer_return_set_flag = True

    # use overlap filter to perform the join.
    overlap_filter = OverlapFilter(tokenizer, threshold, comp_op, allow_missing)
    output_table =  overlap_filter.filter_tables(ltable, rtable,
                                                 l_key_attr, r_key_attr,
                                                 l_join_attr, r_join_attr,
                                                 l_out_attrs, r_out_attrs,
                                                 l_out_prefix, r_out_prefix,
                                                 out_sim_score, n_jobs,
                                                 show_progress)

    # revert the return_set flag of tokenizer, in case it was modified.
    if revert_tokenizer_return_set_flag:
        tokenizer.set_return_set(False)

    return output_table
