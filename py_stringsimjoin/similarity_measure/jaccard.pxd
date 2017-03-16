
from libcpp.vector cimport vector                                               

cdef double jaccard(const vector[int]& tokens1, const vector[int]& tokens2) nogil

