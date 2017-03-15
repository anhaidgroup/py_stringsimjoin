
from libcpp cimport bool                                                        

ctypedef bool (*compfnptr)(double, double) nogil                                

cdef int get_comp_type(comp_op)
cdef compfnptr get_comparison_function(const int comp_type) nogil

cdef int int_min(int a, int b) nogil
cdef int int_max(int a, int b) nogil
