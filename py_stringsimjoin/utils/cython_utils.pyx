
from libcpp cimport bool                                                        

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
