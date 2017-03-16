
from libcpp.vector cimport vector              
from libcpp.pair cimport pair 
from libcpp cimport bool                                                        

cpdef void set_sim_join_cy(                
                                          ltable, rtable, l_attr_index, r_attr_index, tokenizer,         
                                          sim_measure, double threshold, 
                                          comp_op,      
                                          int n_jobs, bool allow_empty,
                                          vector[vector[pair[int, int]]]& output_pairs, 
                                          vector[vector[double]]& output_sim_scores)
