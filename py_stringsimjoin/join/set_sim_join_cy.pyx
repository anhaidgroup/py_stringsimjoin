# set similarity join

import pyprind                                                                  

from cython.parallel import prange                                              
                                                                                
from libc.math cimport ceil, floor, round, sqrt, trunc                          
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair    

from py_stringsimjoin.similarity_measure.cosine cimport cosine                
from py_stringsimjoin.similarity_measure.dice cimport dice                
from py_stringsimjoin.similarity_measure.jaccard cimport jaccard
from py_stringsimjoin.index.position_index_cy cimport PositionIndexCy             
from py_stringsimjoin.utils.cython_utils cimport compfnptr,\
    get_comparison_function, get_comp_type, int_max, int_min, tokenize_lists                       


# Initialize a global variable to keep track of the progress bar                
_progress_bar = None 


cdef void set_sim_join_cy(ltable, rtable, 
                           l_join_attr_index, r_join_attr_index, 
                           tokenizer, sim_measure, double threshold, comp_op,       
                           int n_jobs, bool allow_empty, bool show_progress,
                           vector[vector[pair[int, int]]]& output_pairs, 
                           vector[vector[double]]& output_sim_scores):      
                     
    cdef vector[vector[int]] ltokens, rtokens
    tokenize_lists(ltable, rtable, l_join_attr_index, r_join_attr_index, 
                   tokenizer, ltokens, rtokens)
  
    cdef vector[pair[int, int]] partitions                                      
    cdef int i, n=rtokens.size(), partition_size, start=0, end    
    cdef int sim_type, comp_op_type                                             

    sim_type = get_sim_type(sim_measure)                                        
    comp_op_type = get_comp_type(comp_op)     

    index = build_position_index(ltokens, sim_type, threshold, allow_empty)                            

    partition_size = <int>(<float> n / <float> n_jobs)                           
    for i in xrange(n_jobs):                                                      
        end = start + partition_size                                            
        if end > n or i == n_jobs - 1:                                           
            end = n                                                             
        partitions.push_back(pair[int, int](start, end))                        
        start = end                                                             
        output_pairs.push_back(vector[pair[int, int]]())                        
        output_sim_scores.push_back(vector[double]())                           

    # If the show_progress flag is enabled, then create a new progress bar and  
    # assign it to the global variable.                                         
    if show_progress:                                                           
        global _progress_bar                                                    
        _progress_bar = pyprind.ProgBar(partition_size)   
                                                                                
    for i in prange(n_jobs, nogil=True):                                         
        set_sim_join_partition(partitions[i], ltokens, rtokens, sim_type, 
                               comp_op_type, threshold, allow_empty,
                               index.index, index.size_vector, index.l_empty_ids,
                               index.min_len, index.max_len, 
                               output_pairs[i], output_sim_scores[i], 
                               i, show_progress)                        


cdef void set_sim_join_partition(pair[int, int] partition,                  
                                 vector[vector[int]]& ltokens,                       
                                 vector[vector[int]]& rtokens,                       
                                 int sim_type, int comp_op_type,                                      
                                 double threshold, bool allow_empty, 
                                 omap[int, vector[pair[int, int]]]& index,
                                 vector[int]& size_vector,
                                 vector[int]& l_empty_ids,
                                 int min_len, int max_len,            
                                 vector[pair[int, int]]& output_pairs,               
                                 vector[double]& output_sim_scores,
                                 int thread_id, bool show_progress) nogil:           
    cdef omap[int, int] candidate_overlap, overlap_threshold_cache              
    cdef vector[pair[int, int]] candidates                                      
    cdef vector[int] tokens                                                     
    cdef pair[int, int] cand, entry                                             
    cdef int k=0, j=0, m, i, prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound                           
    cdef double sim_score, overlap_score                                        
    cdef fnptr sim_fn                                           
    cdef compfnptr comp_fn                
    sim_fn = get_sim_function(sim_type)                                         
    comp_fn = get_comparison_function(comp_op_type)
                                                                            
    for i in range(partition.first, partition.second):                          
        tokens = rtokens[i]                                                     
        m = tokens.size()                                                       

        if allow_empty and m == 0:
            for j in l_empty_ids:
                output_pairs.push_back(pair[int, int](j, i))      
                output_sim_scores.push_back(1.0)  
            continue
 
        prefix_length = get_prefix_length(m, sim_type, threshold)               
        size_lower_bound = int_max(get_size_lower_bound(m, sim_type, threshold),
                                   min_len)                               
        size_upper_bound = int_min(get_size_upper_bound(m, sim_type, threshold),
                                   max_len)                               

        for size in range(size_lower_bound, size_upper_bound + 1):              
            overlap_threshold_cache[size] = get_overlap_threshold(size, m, sim_type, threshold)

        for j in range(prefix_length):                                          
            if index.find(tokens[j]) == index.end():                
                continue                                                        
            candidates = index[tokens[j]]                                 
            for cand in candidates:                                             
                current_overlap = candidate_overlap[cand.first]                 
                if current_overlap != -1:                                       
                    cand_num_tokens = size_vector[cand.first]             
                                                                                
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
                    output_pairs.push_back(pair[int, int](entry.first, i))      
                    output_sim_scores.push_back(sim_score)                      

        candidate_overlap.clear()                                               
        overlap_threshold_cache.clear()          

        # If the show_progress flag is enabled, we update the progress bar.     
        # Note that only one of the threads will update the progress bar. To    
        # do so, it releases GIL and updates the global variable that keeps     
        # track of the progress bar.                                            
        if thread_id == 0 and show_progress:                                    
            with gil:                                                           
                global _progress_bar                                            
                _progress_bar.update()   


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
        for j in range(prefix_length):                                          
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

