# edit distance join

import os
import datetime
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer

# This the name of the outfile created, in case the user does not provide any value to the output_file_path argument.
default_output_file_name = "py_stringsimjoin_edit_distance_output"
time_string = datetime.datetime.now().strftime("%H:%M:%S:%f")[0:11]
default_output_file_path = os.path.join(os.getcwd(), default_output_file_name + "_" + time_string + ".csv")

def disk_edit_distance_join(ltable, rtable,
                            l_key_attr, r_key_attr,
                            l_join_attr, r_join_attr,
                            threshold, data_limit=1000000,
                            comp_op='<=', allow_missing=False,
                            l_out_attrs=None, r_out_attrs=None,
                            l_out_prefix='l_', r_out_prefix='r_',
                            out_sim_score=True, n_jobs=-1,
                            show_progress=True, tokenizer=QgramTokenizer(qval=2),
                            temp_dir=os.getcwd(), output_file_path=default_output_file_path):

    """
    WARNING: THIS IS AN EXPERIMENTAL COMMAND. THIS COMMAND IS NOT TESTED. 
    USE AT YOUR OWN RISK.

    Join two tables using edit distance measure.

    This is the disk version of the previous edit_distance_join api.
    There can be a scenario that while performing join on large datasets,
    the intermediate in-memory data structures grow very large and thus lead
    to termination of the program due to insufficient memory. Keeping this problem
    in mind, disk_edit_distance_join is the updated version of the older
    edit_distance_join function that solves the above mentioned problem.
    So if the analysis is being done on the machine with small memory limits or
    if the input tables are too large, then this new disk_edit_distance_join can be
    used to avoid memory exceeding problem while processing.


    It Finds tuple pairs from left table and right table such that the edit
    distance between the join attributes satisfies the condition on input
    threshold. For example, if the comparison operator is '<=', finds tuple
    pairs whose edit distance between the strings that are the values of
    the join attributes is less than or equal to the input threshold, as
    specified in "threshold".

    Note:
        Currently, this method only computes an approximate join result. This is
        because, to perform the join we transform an edit distance measure
        between strings into an overlap measure between qgrams of the strings.
        Hence, we need at least one qgram to be in common between two input
        strings, to appear in the join output. For smaller strings, where all
        qgrams of the strings differ, we cannot process them.

        This method implements a simplified version of the algorithm proposed in
        `Ed-Join: An Efficient Algorithm for Similarity Joins With Edit Distance
        Constraints (Chuan Xiao, Wei Wang and Xuemin Lin), VLDB 08
        <http://www.vldb.org/pvldb/1/1453957.pdf>`_.

    Args:
        ltable (DataFrame): left input table.

        rtable (DataFrame): right input table.

        l_key_attr (string): key attribute in left table.

        r_key_attr (string): key attribute in right table.

        l_join_attr (string): join attribute in left table.

        r_join_attr (string): join attribute in right table.

        threshold (float): edit distance threshold to be satisfied.

        data_limit (int): threshold value for number of rows that would be kept
            in memory before writing the output on the disk. This is the maximum sum
            total of all rows that can be present in memory across all processes at
            a time. This is a new argument compared to edit distance join.
            (defaults to 1M)

        comp_op (string): comparison operator. Supported values are '<=', '<'
            and '=' (defaults to '<=').

        allow_missing (boolean): flag to indicate whether tuple pairs with
            missing value in at least one of the join attributes should be
            included in the output (defaults to False). If this flag is set to
            True, a tuple in ltable with missing value in the join attribute
            will be matched with every tuple in rtable and vice versa.

        l_out_attrs (list): list of attribute names from the left table to be
            included in the output table (defaults to None).

        r_out_attrs (list): list of attribute names from the right table to be
            included in the output table (defaults to None).

        l_out_prefix (string): prefix to be used for the attribute names coming
            from the left table, in the output table (defaults to 'l\_').

        r_out_prefix (string): prefix to be used for the attribute names coming
            from the right table, in the output table (defaults to 'r\_').

        out_sim_score (boolean): flag to indicate whether the edit distance
            score should be included in the output table (defaults to True).
            Setting this flag to True will add a column named '_sim_score' in
            the output table. This column will contain the edit distance scores
            for the tuple pairs in the output.

        n_jobs (int): number of parallel jobs to use for the computation
            (defaults to 1). If -1 is given, all CPUs are used. If 1 is given,
            no parallel computing code is used at all, which is useful for
            debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used
            (where n_cpus is the total number of CPUs in the machine). Thus for
            n_jobs = -2, all CPUs but one are used. If (n_cpus + 1 + n_jobs)
            becomes less than 1, then no parallel computing code will be used
            (i.e., equivalent to the default).

        show_progress (boolean): flag to indicate whether task progress should
            be displayed to the user (defaults to True).

        tokenizer (Tokenizer): tokenizer to be used to tokenize the join
            attributes during filtering, when edit distance measure is
            transformed into an overlap measure. This must be a q-gram tokenizer
            (defaults to 2-gram tokenizer).

        temp_dir (string): absolute path where all the intermediate files will be generated.
            This is a new argument compared to edit distance join. (defaults to the current
            working directory).

        output_file_path (string): absolute path where the output file will be generated.
            Older file with same path and name will be removed. This is a new argument compared
            to edit distance join. (defaults to the current working directory/$default_output_file_name).

    Returns:
        Returns the status of the computation. True if successfully completed else False (boolean).
    """

    from py_stringsimjoin import __use_cython__ 
    if __use_cython__:
        from py_stringsimjoin.join.disk_edit_distance_join_cy import disk_edit_distance_join_cy
        return disk_edit_distance_join_cy(ltable, rtable,
                                          l_key_attr, r_key_attr,
                                          l_join_attr, r_join_attr,
                                          threshold, data_limit,
                                          comp_op,
                                          allow_missing,
                                          l_out_attrs, r_out_attrs,
                                          l_out_prefix, r_out_prefix,
                                          out_sim_score, n_jobs,
                                          show_progress,tokenizer,
                                          temp_dir, output_file_path)
    else:
        raise AssertionError('Cython not installed.')
