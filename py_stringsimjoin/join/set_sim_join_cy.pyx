# set similarity join

import pandas as pd
import pyprind                                                                  

from libc.math cimport ceil, floor, round, sqrt, trunc                          
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair    

from py_stringsimjoin.utils.generic_helper import find_output_attribute_indices,\
    get_output_header_from_tables, get_output_row_from_tables
from py_stringsimjoin.similarity_measure.cosine cimport cosine                
from py_stringsimjoin.similarity_measure.dice cimport dice                
from py_stringsimjoin.similarity_measure.jaccard cimport jaccard
from py_stringsimjoin.index.position_index_cy cimport PositionIndexCy             
from py_stringsimjoin.utils.cython_utils cimport compfnptr,\
    get_comparison_function, get_comp_type, int_max, int_min, tokenize_lists                       


def set_sim_join_cy(ltable, rtable,                   
                    l_columns, r_columns,                       
                    l_key_attr, r_key_attr,                     
                    l_join_attr, r_join_attr,                   
                    tokenizer, sim_measure, threshold, comp_op,              
                    allow_empty,                                
                    l_out_attrs, r_out_attrs,                   
                    l_out_prefix, r_out_prefix,                 
                    out_sim_score, show_progress):

    # find column indices of key attr and output attrs in ltable                
    l_key_attr_index = l_columns.index(l_key_attr)                              
    l_join_attr_index = l_columns.index(l_join_attr)                            
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs) 
                                                                                
    # find column indices of key attr and output attrs in rtable                
    r_key_attr_index = r_columns.index(r_key_attr)                              
    r_join_attr_index = r_columns.index(r_join_attr)                            
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs) 

                   
    cdef vector[vector[int]] ltokens, rtokens
    tokenize_lists(ltable, rtable, l_join_attr_index, r_join_attr_index, 
                   tokenizer, ltokens, rtokens)

    cdef int sim_type
    sim_type = get_sim_type(sim_measure)

    cdef PositionIndexCy index = PositionIndexCy()                                
    index = build_position_index(ltokens, sim_type, threshold, allow_empty)     

    cdef omap[int, int] candidate_overlap, overlap_threshold_cache              
    cdef vector[pair[int, int]] candidates                                      
    cdef vector[int] tokens                                                     
    cdef pair[int, int] cand, entry                                             
    cdef int k, j, m, i, prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound                           
    cdef double sim_score, overlap_score                                        
    cdef fnptr sim_fn                                           
    cdef compfnptr comp_fn               
    sim_fn = get_sim_function(sim_type)                                         
    comp_fn = get_comparison_function(get_comp_type(comp_op))

    output_rows = []                                                            
    has_output_attributes = (l_out_attrs is not None or                         
                             r_out_attrs is not None)                           
                                                                                
    if show_progress:                                                           
        prog_bar = pyprind.ProgBar(len(rtable))
                                                                            
    for i in range(rtokens.size()):                        
        tokens = rtokens[i]                                                     
        m = tokens.size()                                                    

        if allow_empty and m == 0:
            for j in index.l_empty_ids:
                if has_output_attributes:                                       
                    output_row = get_output_row_from_tables(                    
                                     ltable[j], rtable[i],                     
                                     l_key_attr_index, r_key_attr_index,        
                                     l_out_attrs_indices,                       
                                     r_out_attrs_indices)                       
                else:                                                           
                    output_row = [ltable[j][l_key_attr_index],             
                                  rtable[i][r_key_attr_index]]                      
                                                                                
                if out_sim_score:                                               
                    output_row.append(1.0)                                      
                output_rows.append(output_row)   
            continue
 
        prefix_length = get_prefix_length(m, sim_type, threshold)               
        size_lower_bound = int_max(get_size_lower_bound(m, sim_type, threshold),
                                   index.min_len)                               
        size_upper_bound = int_min(get_size_upper_bound(m, sim_type, threshold),
                                   index.max_len)                         

        for size in range(size_lower_bound, size_upper_bound + 1):              
            overlap_threshold_cache[size] = get_overlap_threshold(size, m, sim_type, threshold)

        for j in range(min(m, prefix_length)):                                          
            if index.index.find(tokens[j]) == index.index.end():                
                continue                                                        
            candidates = index.index[tokens[j]]                                 
            for cand in candidates:                                             
                current_overlap = candidate_overlap[cand.first]                 
                if current_overlap != -1:                                       
                    cand_num_tokens = index.size_vector[cand.first]             
                                                                                
                    # only consider candidates satisfying the size filter       
                    # condition.                                                
                    if size_lower_bound <= cand_num_tokens <= size_upper_bound: 
                                                                                
                        if m - j <= cand_num_tokens - cand.second:              
                            overlap_upper_bound = m - j                         
                        else:                                                   
                            overlap_upper_bound = cand_num_tokens - cand.second 

                        # only consider candidates for which the overlap upper  
                        # bound is at least the required overlap.               
                        if (current_overlap + overlap_upper_bound >=            
                                overlap_threshold_cache[cand_num_tokens]):      
                            candidate_overlap[cand.first] = current_overlap + 1 
                        else:                                                   
                            candidate_overlap[cand.first] = -1                  
                                                                                
        for entry in candidate_overlap:                                         
            if entry.second > 0:                                                
                sim_score = sim_fn(ltokens[entry.first], tokens)                

                if comp_fn(sim_score, threshold):                                       
                    if has_output_attributes:                                       
                        output_row = get_output_row_from_tables(                    
                                     ltable[entry.first], rtable[i],           
                                     l_key_attr_index, r_key_attr_index,        
                                     l_out_attrs_indices, r_out_attrs_indices)  
                    else:                                                           
                        output_row = [ltable[entry.first][l_key_attr_index],   
                                      rtable[i][r_key_attr_index]]                      
                                                                                
                    # if out_sim_score flag is set, append the overlap coefficient  
                    # score to the output record.                                   
                    if out_sim_score:                                               
                        output_row.append(sim_score)                                
                                                                                
                    output_rows.append(output_row)  

        candidate_overlap.clear()                                               
        overlap_threshold_cache.clear()          

        if show_progress:                                                       
            prog_bar.update()

    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,       
                                                  l_out_attrs, r_out_attrs,     
                                                  l_out_prefix, r_out_prefix)   
    if out_sim_score:                                                           
        output_header.append("_sim_score")                                      
                                                                                
    output_table = pd.DataFrame(output_rows, columns=output_header)             
    return output_table   

cdef PositionIndexCy build_position_index(vector[vector[int]]& token_vectors, 
                               int& sim_type, double& threshold, 
                               bool allow_empty):
    cdef PositionIndexCy pos_index = PositionIndexCy()
    cdef vector[int] tokens, size_vector                                        
    cdef int prefix_length, token, i, j, m, n=token_vectors.size(), min_len=100000, max_len=0
    cdef omap[int, vector[pair[int, int]]] index
    cdef vector[int] empty_l_ids                               
    for i in range(n):                                                         
        tokens = token_vectors[i]                                           
        m = tokens.size()                                                     
        size_vector.push_back(m)                                                
        prefix_length = get_prefix_length(m, sim_type, threshold)         
        for j in range(min(m, prefix_length)):                                          
            index[tokens[j]].push_back(pair[int, int](i, j))                  
        if m > max_len:                                                         
            max_len = m                                                         
        if m < min_len:                                                         
            min_len = m
        if allow_empty and m == 0:
            empty_l_ids.push_back(i)

    pos_index.set_fields(index, size_vector, empty_l_ids, 
                         min_len, max_len, threshold)
    return pos_index      


cdef int get_prefix_length(int& num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE                                                  
        return <int>(num_tokens - ceil(threshold * threshold * num_tokens) + 1.0)
    elif sim_type == 1: # DICE                                                  
        return <int>(num_tokens - ceil((threshold / (2 - threshold)) * num_tokens) + 1.0)
    elif sim_type == 2: # JACCARD:                                              
        return <int>(num_tokens - ceil(threshold * num_tokens) + 1.0)           
                                                                                
cdef int get_size_lower_bound(int& num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE                                                  
        return <int>ceil(threshold * threshold * num_tokens)                    
    elif sim_type == 1: # DICE                                                  
        return <int>ceil((threshold / (2 - threshold)) * num_tokens)            
    elif sim_type == 2: # JACCARD:                                              
        return <int>ceil(threshold * num_tokens)                                
                                                                                
cdef int get_size_upper_bound(int& num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE                                                  
        return <int>floor(num_tokens / (threshold * threshold))                 
    elif sim_type == 1: # DICE                                                  
        return <int>floor(((2 - threshold) / threshold) * num_tokens)           
    elif sim_type == 2: # JACCARD:                                              
        return <int>floor(num_tokens / threshold)                              
                                                                                
cdef int get_overlap_threshold(int& l_num_tokens, int& r_num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE                                                  
        return <int>ceil(threshold * sqrt(<double>(l_num_tokens * r_num_tokens)))
    elif sim_type == 1: # DICE                                                  
        return <int>ceil((threshold / 2) * (l_num_tokens + r_num_tokens))       
    elif sim_type == 2: # JACCARD:                                              
        return <int>ceil((threshold / (1 + threshold)) * (l_num_tokens + r_num_tokens))
                                                                                
ctypedef double (*fnptr)(const vector[int]&, const vector[int]&) nogil          
                                                                                
cdef fnptr get_sim_function(int& sim_type) nogil:                               
    if sim_type == 0: # COSINE                                                  
        return cosine                                                           
    elif sim_type == 1: # DICE                                                  
        return dice                                                             
    elif sim_type == 2: # JACCARD:                                              
        return jaccard 

cdef int get_sim_type(sim_measure):                          
    if sim_measure == 'COSINE': # COSINE                                                  
        return 0                                                                
    elif sim_measure == 'DICE': # DICE                                                  
        return 1                                                                
    elif sim_measure == 'JACCARD': # JACCARD:                                              
        return 2                                                                

