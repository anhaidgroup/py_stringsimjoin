
from libcpp cimport bool                                                        
from libcpp.vector cimport vector
from libcpp.pair cimport pair  

from py_stringsimjoin.index.inverted_index_cy cimport InvertedIndexCy           


ctypedef bool (*compfnptr)(double, double) nogil                                

cdef void tokenize_lists(ltable, rtable,                                        
                         l_join_attr_index, r_join_attr_index,                  
                         tokenizer,                                             
                         vector[vector[int]]& ltokens,                          
                         vector[vector[int]]& rtokens)

cdef generate_output_table(ltable_array, rtable_array,                     
                           vector[vector[pair[int, int]]]& output_pairs,   
                           vector[vector[double]]& output_sim_scores,      
                           l_key_attr_index, r_key_attr_index,             
                           l_out_attrs_indices, r_out_attrs_indices,       
                           out_sim_score, output_header, n_jobs)

cdef void build_inverted_index(vector[vector[int]]& token_vectors,              
                               InvertedIndexCy inv_index)

cdef int get_comp_type(comp_op)
cdef compfnptr get_comparison_function(const int comp_type) nogil

cdef int int_min(int a, int b) nogil
cdef int int_max(int a, int b) nogil
