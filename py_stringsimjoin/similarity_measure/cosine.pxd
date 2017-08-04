
from libcpp.vector cimport vector                                               

cdef double cosine(const vector[int]& tokens1, const vector[int]& tokens2) nogil

