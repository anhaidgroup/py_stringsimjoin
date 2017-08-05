
from libcpp.vector cimport vector                                               
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair   

cdef class PositionIndexCy:
    cdef void set_fields(self, omap[int, vector[pair[int, int]]]& ind, vector[int]& sv, 
                         vector[int]& emp_ids, int min_l, int max_l, double t):
        self.index = ind
        self.size_vector = sv
        self.l_empty_ids = emp_ids
        self.min_len = min_l
        self.max_len = max_l
        self.threshold = t
