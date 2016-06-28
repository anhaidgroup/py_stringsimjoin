from joblib import delayed
from joblib import Parallel
import pandas as pd
import pyprind

from py_stringsimjoin.utils.helper_functions import build_dict_from_table, \
    get_num_processes_to_launch, split_table


class Filter(object):
    """Filter base class.
    """
    def __init__(self, allow_missing=False):
        self.allow_missing = allow_missing

    def filter_candset(self, candset,
                       candset_l_key_attr, candset_r_key_attr,
                       ltable, rtable,
                       l_key_attr, r_key_attr,
                       l_filter_attr, r_filter_attr,
                       n_jobs=1, show_progress=True):
        """Finds candidate matching pairs of strings from the input candset.

        Args:
            candset (dataframe): input candidate set.

            candset_l_key_attr (string): attribute in candidate set that is a key in left table.

            candset_r_key_attr (string): attribute in candidate set that is a key in right table.

            ltable (dataframe): left input table.

            rtable (dataframe): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_filter_attr (string): attribute to be used by the filter, in left table.

            r_filter_attr (string): attribute to be used by the filter,  in right table.

            n_jobs (int): The number of jobs to use for the computation (defaults to 1).                                                                                            
                If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, 
                which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. 
                Thus for n_jobs = -2, all CPUs but one are used. If (n_cpus + 1 + n_jobs) becomes less than 1,
                then n_jobs is set to 1.

            show_progress (boolean): flag to indicate if task progress need to be shown (defaults to True).

        Returns:
            output table (dataframe)
        """

        # check for empty candset
        if candset.empty:
            return candset

        # convert the filter attributes to string type, in case it is int or float.
        revert_l_filter_attr_type = False
        orig_l_filter_attr_type = ltable[l_filter_attr].dtype
        if (orig_l_filter_attr_type == pd.np.int64 or
            orig_l_filter_attr_type == pd.np.float64):
            ltable[l_filter_attr] = ltable[l_filter_attr].astype(str)
            revert_l_filter_attr_type = True

        revert_r_filter_attr_type = False
        orig_r_filter_attr_type = rtable[r_filter_attr].dtype
        if (orig_r_filter_attr_type == pd.np.int64 or
            orig_r_filter_attr_type == pd.np.float64):
            rtable[r_filter_attr] = rtable[r_filter_attr].astype(str)
            revert_r_filter_attr_type = True

        # computes the actual number of jobs to launch.
        n_jobs = get_num_processes_to_launch(n_jobs)

        if n_jobs == 1:
            output_table =  _filter_candset_split(candset,
                                         candset_l_key_attr, candset_r_key_attr,
                                         ltable, rtable,
                                         l_key_attr, r_key_attr,
                                         l_filter_attr, r_filter_attr,
                                         self, show_progress)
        else:
            candset_splits = split_table(candset, n_jobs)
            results = Parallel(n_jobs=n_jobs)(delayed(_filter_candset_split)(
                                      candset_splits[job_index],
                                      candset_l_key_attr, candset_r_key_attr,
                                      ltable, rtable,
                                      l_key_attr, r_key_attr,
                                      l_filter_attr, r_filter_attr,
                                      self,
                                      (show_progress and (job_index==n_jobs-1)))
                                          for job_index in range(n_jobs))
            output_table = pd.concat(results)

        # revert the type of filter attributes to their original type, in case
        # it was converted to string type.
        if revert_l_filter_attr_type:
            ltable[l_filter_attr] = ltable[l_filter_attr].astype(
                                                        orig_l_filter_attr_type)

        if revert_r_filter_attr_type:
            rtable[r_filter_attr] = rtable[r_filter_attr].astype(
                                                        orig_r_filter_attr_type)

        return output_table        


def _filter_candset_split(candset,
                          candset_l_key_attr, candset_r_key_attr,
                          ltable, rtable,
                          l_key_attr, r_key_attr,
                          l_filter_attr, r_filter_attr,
                          filter_object, show_progress):
    # Find column indices of key attr and filter attr in ltable
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    l_filter_attr_index = l_columns.index(l_filter_attr)

    # Find column indices of key attr and filter attr in rtable
    r_columns = list(rtable.columns.values)
    r_key_attr_index = r_columns.index(r_key_attr)
    r_filter_attr_index = r_columns.index(r_filter_attr)
    
    # Build a dictionary on ltable
    ltable_dict = build_dict_from_table(ltable, l_key_attr_index,
                                        l_filter_attr_index,
                                        remove_null=False)

    # Build a dictionary on rtable
    rtable_dict = build_dict_from_table(rtable, r_key_attr_index,
                                        r_filter_attr_index,
                                        remove_null=False)

    # Find indices of l_key_attr and r_key_attr in candset
    candset_columns = list(candset.columns.values)
    candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)
    candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)

    valid_rows = []

    if show_progress:
        prog_bar = pyprind.ProgBar(len(candset))

    for candset_row in candset.itertuples(index = False):
        l_id = candset_row[candset_l_key_attr_index]
        r_id = candset_row[candset_r_key_attr_index]

        l_row = ltable_dict[l_id]
        r_row = rtable_dict[r_id]

        valid_rows.append(not filter_object.filter_pair(
                                  l_row[l_filter_attr_index],
                                  r_row[r_filter_attr_index]))

        if show_progress:
            prog_bar.update()

    return candset[valid_rows]
