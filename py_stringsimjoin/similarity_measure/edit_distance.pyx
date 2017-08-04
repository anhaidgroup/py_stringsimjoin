
from libcpp.string cimport string                                               
from libc.stdlib cimport malloc, free

cdef inline int int_min3(int a, int b, int c) nogil:
    if (a<=b) and (a<= c):
        return a
    elif (b<=c):
        return b
    else:
        return c

cdef double edit_distance(const string& str1, const string& str2) nogil:
    cdef int len_str1 = str1.length(), len_str2 = str2.length()

    cdef int ins_cost = 1, del_cost = 1, sub_cost = 1, trans_cost = 1

    cdef int edit_dist, i = 0, j = 0

    if len_str1 == 0:
        return len_str2 * ins_cost

    if len_str2 == 0:
        return len_str1 * del_cost

    cdef int *d_mat = <int*>malloc((len_str1 + 1) * (len_str2 + 1) * sizeof(int)) 

    for i in range(len_str1 + 1):
        d_mat[i*(len_str2 + 1)] = i * del_cost

    for j in range(len_str2 + 1):
        d_mat[j] = j * ins_cost

    cdef unsigned char lchar = 0
    cdef unsigned char rchar = 0

    for i in range(len_str1):
        lchar = str1[i]
        for j in range(len_str2):
            rchar = str2[j]

            d_mat[(i+1)*(len_str2 + 1) + j+1] = int_min3(d_mat[(i + 1)*(len_str2 + 1) + j] + ins_cost,
                                     d_mat[i*(len_str2 + 1) + j + 1] + del_cost,
                                     d_mat[i*(len_str2 + 1) + j] + (sub_cost if lchar != rchar else 0))
    edit_dist = d_mat[len_str1*(len_str2 + 1) + len_str2]
    free(d_mat)
    return <double>edit_dist
