# edit distance join
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer

def edit_distance_join(ltable, rtable,
                       l_key_attr, r_key_attr,
                       l_join_attr, r_join_attr,
                       threshold, comp_op='<=',
                       allow_missing=False,
                       l_out_attrs=None, r_out_attrs=None,
                       l_out_prefix='l_', r_out_prefix='r_',
                       out_sim_score=True, n_jobs=1, show_progress=True,
                       tokenizer=QgramTokenizer(qval=2)):
    from py_stringsimjoin import __use_cython__ 
    if __use_cython__:
        from py_stringsimjoin.join.edit_distance_join_cy import edit_distance_join_cy                     
        return edit_distance_join_cy(ltable, rtable,
                                     l_key_attr, r_key_attr,
                                     l_join_attr, r_join_attr,
                                     threshold, comp_op,
                                     allow_missing,
                                     l_out_attrs, r_out_attrs,
                                     l_out_prefix, r_out_prefix,
                                     out_sim_score, n_jobs, show_progress,
                                     tokenizer)
    else:
        from py_stringsimjoin.join.edit_distance_join_py import edit_distance_join_py
        return edit_distance_join_py(ltable, rtable,
                                     l_key_attr, r_key_attr,
                                     l_join_attr, r_join_attr,
                                     threshold, comp_op,
                                     allow_missing,
                                     l_out_attrs, r_out_attrs,
                                     l_out_prefix, r_out_prefix,
                                     out_sim_score, n_jobs, show_progress,
                                     tokenizer)
