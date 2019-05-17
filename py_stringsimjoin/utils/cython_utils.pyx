
import pandas as pd

from py_stringsimjoin.utils.generic_helper import get_output_row_from_tables
from py_stringsimjoin.utils.token_ordering import gen_token_ordering_for_tables,\
    order_using_token_ordering

from libcpp cimport bool                                                        
from libcpp.vector cimport vector                                               
from libcpp.pair cimport pair  
from libcpp.map cimport map as omap

from py_stringsimjoin.index.inverted_index_cy cimport InvertedIndexCy           


cdef void tokenize_lists(ltable, rtable, 
                         l_join_attr_index, r_join_attr_index, 
                         tokenizer,      
                         vector[vector[int]]& ltokens, 
                         vector[vector[int]]& rtokens): 

    token_ordering = gen_token_ordering_for_tables(                             
                         [ltable, rtable],                                      
                         [l_join_attr_index, r_join_attr_index],                          
                         tokenizer)                                             
                                                                                
    for lrow in ltable:                                                         
        lstr = lrow[l_join_attr_index]                                               
        py_tokens = order_using_token_ordering(                                 
                        tokenizer.tokenize(lstr), token_ordering)
        ltokens.push_back(<vector[int]>py_tokens)                         

    for rrow in rtable:
        rstr = rrow[r_join_attr_index]                                               
        py_tokens = order_using_token_ordering(                                 
                        tokenizer.tokenize(rstr), token_ordering)   
        rtokens.push_back(<vector[int]>py_tokens)         
            

cdef generate_output_table(ltable_array, rtable_array, 
                           vector[vector[pair[int, int]]]& output_pairs, 
                           vector[vector[double]]& output_sim_scores, 
                           l_key_attr_index, r_key_attr_index, 
                           l_out_attrs_indices, r_out_attrs_indices, 
                           out_sim_score, output_header, n_jobs):

    output_rows = []                                                            
    has_output_attributes = (len(l_out_attrs_indices) > 0 or                         
                             len(r_out_attrs_indices) > 0)       

    cdef int i, j                                                               
    cdef pair[int, int] pair_entry                                              
    for i in xrange(n_jobs):                                                    
        for j in xrange(output_pairs[i].size()):                                
            pair_entry = output_pairs[i][j]                                     
            if has_output_attributes:                                           
                output_row = get_output_row_from_tables(                        
                    ltable_array[pair_entry.first], rtable_array[pair_entry.second],
                    l_key_attr_index, r_key_attr_index,                         
                    l_out_attrs_indices, r_out_attrs_indices)                   
            else:                                                               
                output_row = [ltable_array[pair_entry.first][l_key_attr_index], 
                              rtable_array[pair_entry.second][r_key_attr_index]]
                                                                                
            # if out_sim_score flag is set, append the similarity score         
            # to the output record.                                             
            if out_sim_score:                                                   
                output_row.append(output_sim_scores[i][j])                      
                                                                                
            output_rows.append(output_row)  

    # generate a dataframe from the list of output rows                         
    output_table = pd.DataFrame(output_rows, columns=output_header)   

    return output_table


cdef void build_inverted_index(vector[vector[int]]& token_vectors, 
                               InvertedIndexCy inv_index):
    cdef vector[int] tokens, size_vector                                        
    cdef int i, j, m, n=token_vectors.size()                                    
    cdef omap[int, vector[int]] index                                           
    for i in xrange(n):                                                          
        tokens = token_vectors[i]                                               
        m = tokens.size()                                                       
        size_vector.push_back(m)                                                
        for j in range(m):                                                      
            index[tokens[j]].push_back(i)                                       
    inv_index.set_fields(index, size_vector) 


cdef int get_comp_type(comp_op):                                                
    if comp_op == '<':                                                          
        return 0                                                                
    elif comp_op == '<=':                                                       
        return 1                                                                
    elif comp_op == '>':                                                        
        return 2                                                                
    elif comp_op == '>=':                                                       
        return 3                                                                
    elif comp_op == '=':                                                        
        return 4


cdef compfnptr get_comparison_function(const int comp_type) nogil:              
    if comp_type == 0:                                                          
        return lt_compare                                                       
    elif comp_type == 1:                                                        
        return le_compare                                                       
    elif comp_type == 2:                                                        
        return gt_compare                                                       
    elif comp_type == 3:                                                        
        return ge_compare                                                       
    elif comp_type == 4:                                                        
        return eq_compare 


cdef bool eq_compare(double val1, double val2) nogil:                           
    return val1 == val2                                                         

                                                                                
cdef bool le_compare(double val1, double val2) nogil:                           
    return val1 <= val2                                                         

                                                                                
cdef bool lt_compare(double val1, double val2) nogil:                           
    return val1 < val2                                                          

                                                                                
cdef bool ge_compare(double val1, double val2) nogil:                           
    return val1 >= val2                                                         

                                                                                
cdef bool gt_compare(double val1, double val2) nogil:                           
    return val1 > val2    


cdef int int_min(int a, int b) nogil:                                           
    return a if a <= b else b                                                   

                                                                                
cdef int int_max(int a, int b) nogil:                                           
    return a if a >= b else b    
