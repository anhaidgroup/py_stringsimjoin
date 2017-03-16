
from libc.math cimport sqrt                                                     
from libcpp.vector cimport vector 

cdef double cosine(const vector[int]& tokens1, const vector[int]& tokens2) nogil:
    cdef int i=0, j=0, size1 = tokens1.size(), size2 = tokens2.size()           
    cdef int sum_of_size = size1 + size2                                        
    if sum_of_size == 0:                                                        
        return 1.0                                                              
    if size1 == 0 or size2 == 0:                                                
        return 0.0                                                              
    cdef int overlap = 0                                                        
    while i < size1 and j < size2:                                              
        if tokens1[i] == tokens2[j]:                                            
            overlap += 1                                                        
            i += 1                                                              
            j += 1                                                              
        elif tokens1[i] < tokens2[j]:                                           
            i += 1                                                              
        else:                                                                   
            j += 1                                                              
    return <double>overlap / sqrt(<double>(size1*size2))    
