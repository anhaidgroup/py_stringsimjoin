
from libcpp.vector cimport vector                                               
from libcpp.map cimport map as omap  

cdef class InvertedIndexCy:
    cdef void set_fields(self, omap[int, vector[int]]& ind, vector[int]& sv):
        self.index = ind
        self.size_vector = sv

    cdef void build_prefix_index(self, vector[vector[int]]& token_vectors, int qval, double threshold):
        cdef int ii, jj, m, n, prefix_length
        cdef vector[int] tokens

        n = token_vectors.size()
        for ii in range(n):
            tokens = token_vectors[ii]
            m = tokens.size()
            self.size_vector.push_back(m)
            prefix_length = min(int(qval * threshold + 1), m)
            
            for jj in range(prefix_length):
                self.index[tokens[jj]].push_back(ii)
 
