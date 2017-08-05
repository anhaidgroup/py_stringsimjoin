
from libcpp.vector cimport vector                                               
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair                                                   

'''
cdef extern from "position_index_cy.h" nogil:                                      
    cdef cppclass PositionIndexCy nogil:                                          
        PositionIndexCy()                                                         
        PositionIndexCy(omap[int, vector[pair[int, int]]]&, vector[int]&, vector[int]&, int&, int&, double&)
        void set_fields(omap[int, vector[pair[int, int]]]&, vector[int]&, vector[int]&, int&, int&, double&)
        omap[int, vector[pair[int, int]]] index                                 
        int min_len, max_len                                                    
        vector[int] size_vector, l_empty_ids                                                 
        double threshold 
'''

cdef class PositionIndexCy:                                        
    cdef void set_fields(self, omap[int, vector[pair[int, int]]]&, vector[int]&, vector[int]&, int, int, double)
    cdef omap[int, vector[pair[int, int]]] index                                 
    cdef int min_len, max_len                                                    
    cdef vector[int] size_vector, l_empty_ids
    cdef double threshold   
