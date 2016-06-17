# overlap join
from py_stringsimjoin.filter.overlap_filter import OverlapFilter

def overlap_join(ltable, rtable,
                 l_key_attr, r_key_attr,
                 l_join_attr, r_join_attr,
                 tokenizer,
                 threshold,
                 l_out_attrs=None, r_out_attrs=None,
                 l_out_prefix='l_', r_out_prefix='r_',
                 out_sim_score=True, n_jobs=1):
    """Join two tables using overlap similarity measure.

    Finds tuple pairs from ltable and rtable such that
    Overlap(ltable.l_join_attr, rtable.r_join_attr) >= threshold

    Args:
    ltable, rtable : Pandas data frame
    l_key_attr, r_key_attr : String, key attribute from ltable and rtable
    l_join_attr, r_join_attr : String, join attribute from ltable and rtable
    tokenizer : Tokenizer object, tokenizer to be used to tokenize join attributes
    threshold : float, overlap threshold to be satisfied
    l_out_attrs, r_out_attrs : list of attributes to be included in the output table from ltable and rtable
    l_out_prefix, r_out_prefix : String, prefix to be used in the attribute names of the output table
    out_sim_score : boolean, indicates if similarity score needs to be included in the output table

    Returns:
    result : Pandas data frame
    """
    overlap_filter = OverlapFilter(tokenizer, threshold)
    return overlap_filter.filter_tables(ltable, rtable,
                                        l_key_attr, r_key_attr,
                                        l_join_attr, r_join_attr,
                                        l_out_attrs, r_out_attrs,
                                        l_out_prefix, r_out_prefix,
                                        out_sim_score, n_jobs)
