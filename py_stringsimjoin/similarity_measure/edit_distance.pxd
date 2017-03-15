
from libcpp.string cimport string

cdef double edit_distance(const string& str1, const string& str2) nogil
