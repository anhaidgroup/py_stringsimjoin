# edit distance join
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from os import getcwd
import os

final_output_file_name = "py_stringsimjoin_edit_distance_output.csv"
def edit_distance_join_disk(ltable, rtable,
                            l_key_attr, r_key_attr,
                            l_join_attr, r_join_attr,
                            threshold, data_limit = 100000,
                            comp_op='<=',
                            allow_missing=False,
                            l_out_attrs=None, r_out_attrs=None,
                            l_out_prefix='l_', r_out_prefix='r_',
                            out_sim_score=True, n_jobs=1, show_progress=True,
                            tokenizer=QgramTokenizer(qval=2), temp_file_path = getcwd(), output_file_path = os.path.join(getcwd(),final_output_file_name)):
    from py_stringsimjoin import __use_cython__ 
    if __use_cython__:
        from py_stringsimjoin.join.edit_distance_join_disk_cy import edit_distance_join_disk_cy    
        return edit_distance_join_disk_cy(ltable, rtable,
                                          l_key_attr, r_key_attr,
                                          l_join_attr, r_join_attr,
                                          threshold, data_limit,
                                          comp_op,
                                          allow_missing,
                                          l_out_attrs, r_out_attrs,
                                          l_out_prefix, r_out_prefix,
                                          out_sim_score, n_jobs, show_progress,
                                          tokenizer, temp_file_path, output_file_path)
    else:
      print("Cython not present")