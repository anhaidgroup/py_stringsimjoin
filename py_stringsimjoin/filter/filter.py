from joblib import delayed, Parallel
import pandas as pd
import pyprind

from py_stringsimjoin.utils.generic_helper import build_dict_from_table, \
    get_num_processes_to_launch, split_table
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_key_attr, validate_input_table


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
        """Finds candidate matching pairs of strings from the input candidate 
        set.

        Args:
            candset (DataFrame): input candidate set.

            candset_l_key_attr (string): attribute in candidate set which is a 
                key in left table.

            candset_r_key_attr (string): attribute in candidate set which is a 
                key in right table.

            ltable (DataFrame): left input table.

            rtable (DataFrame): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_filter_attr (string): attribute in left table on which the filter 
                should be applied.                                              
                                                                                
            r_filter_attr (string): attribute in right table on which the filter
                should be applied.

            n_jobs (int): number of parallel jobs to use for the computation    
                (defaults to 1). If -1 is given, all CPUs are used. If 1 is     
                given, no parallel computing code is used at all, which is      
                useful for debugging. For n_jobs below -1,                      
                (n_cpus + 1 + n_jobs) are used (where n_cpus is the total       
                number of CPUs in the machine). Thus for n_jobs = -2, all CPUs  
                but one are used. If (n_cpus + 1 + n_jobs) becomes less than 1, 
                then no parallel computing code will be used (i.e., equivalent  
                to the default).
                                                                                
            show_progress (boolean): flag to indicate whether task progress     
                should be displayed to the user (defaults to True). 

        Returns:
            An output table containing tuple pairs from the candidate set that 
            survive the filter (DataFrame).
        """

        # check if the input candset is a dataframe
        validate_input_table(candset, 'candset')

        # check if the candset key attributes exist
        validate_attr(candset_l_key_attr, candset.columns,
                      'left key attribute', 'candset')
        validate_attr(candset_r_key_attr, candset.columns,
                      'right key attribute', 'candset')

        # check if the input tables are dataframes
        validate_input_table(ltable, 'left table')
        validate_input_table(rtable, 'right table')

        # check if the key attributes filter join attributes exist
        validate_attr(l_key_attr, ltable.columns,
                      'key attribute', 'left table')
        validate_attr(r_key_attr, rtable.columns,
                      'key attribute', 'right table')
        validate_attr(l_filter_attr, ltable.columns,
                      'filter attribute', 'left table')
        validate_attr(r_filter_attr, rtable.columns,
                      'filter attribute', 'right table')

        # check if the filter attributes are not of numeric type                      
        validate_attr_type(l_filter_attr, ltable[l_filter_attr].dtype,          
                           'filter attribute', 'left table')                    
        validate_attr_type(r_filter_attr, rtable[r_filter_attr].dtype,          
                           'filter attribute', 'right table')

        # check if the key attributes are unique and do not contain 
        # missing values
        validate_key_attr(l_key_attr, ltable, 'left table')
        validate_key_attr(r_key_attr, rtable, 'right table')

        # check for empty candset
        if candset.empty:
            return candset

        # Do a projection on the input dataframes to keep only required 
        # attributes. Note that this does not create a copy of the dataframes. 
        # It only creates a view on original dataframes.
        ltable_projected = ltable[[l_key_attr, l_filter_attr]]
        rtable_projected = rtable[[r_key_attr, r_filter_attr]]

        # computes the actual number of jobs to launch.
        n_jobs = min(get_num_processes_to_launch(n_jobs), len(candset))
        
        if n_jobs <= 1:
            # if n_jobs is 1, do not use any parallel code.                     
            output_table =  _filter_candset_split(candset,
                                         candset_l_key_attr, candset_r_key_attr,
                                         ltable_projected, rtable_projected,
                                         l_key_attr, r_key_attr,
                                         l_filter_attr, r_filter_attr,
                                         self, show_progress)
        else:
            # if n_jobs is above 1, split the candset into n_jobs splits and    
            # filter each candset split in a separate process.
            candset_splits = split_table(candset, n_jobs)
            results = Parallel(n_jobs=n_jobs)(delayed(_filter_candset_split)(
                                      candset_splits[job_index],
                                      candset_l_key_attr, candset_r_key_attr,
                                      ltable_projected, rtable_projected,
                                      l_key_attr, r_key_attr,
                                      l_filter_attr, r_filter_attr,
                                      self,
                                      (show_progress and (job_index==n_jobs-1)))
                                          for job_index in range(n_jobs))
            output_table = pd.concat(results)

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
