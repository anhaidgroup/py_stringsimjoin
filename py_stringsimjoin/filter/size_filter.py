# Size Filter

from joblib import delayed, Parallel
from six.moves import xrange
import pandas as pd
import pyprind

from py_stringsimjoin.filter.filter import Filter
from py_stringsimjoin.filter.filter_utils import get_size_lower_bound, \
                                                 get_size_upper_bound
from py_stringsimjoin.index.size_index import SizeIndex
from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_array, \
    find_output_attribute_indices, get_attrs_to_project, \
    get_num_processes_to_launch, get_output_header_from_tables, \
    get_output_row_from_tables, remove_redundant_attrs, split_table
from py_stringsimjoin.utils.missing_value_handler import \
    get_pairs_with_missing_value
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_key_attr, validate_input_table, \
    validate_threshold, validate_tokenizer_for_sim_measure, \
    validate_output_attrs, validate_sim_measure_type


class SizeFilter(Filter):
    """Finds candidate matching pairs of strings using size filtering technique.

    For similarity measures such as cosine, Dice, Jaccard and overlap, the 
    filter finds candidate string pairs that may have similarity score greater 
    than or equal to the input threshold, as specified in "threshold". For 
    distance measures such as edit distance, the filter finds candidate string 
    pairs that may have distance score less than or equal to the threshold.

    To know more about size filtering, refer to the `string matching chapter 
    <http://pages.cs.wisc.edu/~anhai/py_stringmatching/dibook-string-matching.pdf>`_ 
    of the "Principles of Data Integration" book.

    Args:
        tokenizer (Tokenizer): tokenizer to be used.
        sim_measure_type (string): similarity measure type. Supported types are 
            'JACCARD', 'COSINE', 'DICE', 'OVERLAP' and 'EDIT_DISTANCE'.
        threshold (float): threshold to be used by the filter.
        allow_empty (boolean): A flag to indicate whether pairs in which both 
            strings are tokenized into an empty set of tokens should 
            survive the filter (defaults to True). This flag is not valid for 
            measures such as 'OVERLAP' and 'EDIT_DISTANCE'. 
        allow_missing (boolean): A flag to indicate whether pairs containing 
            missing value should survive the filter (defaults to False).

    Attributes:
        tokenizer (Tokenizer): An attribute to store the tokenizer.
        sim_measure_type (string): An attribute to store the similarity measure 
            type.
        threshold (float): An attribute to store the threshold value.
        allow_empty (boolean): An attribute to store the value of the flag    
            allow_empty.
        allow_missing (boolean): An attribute to store the value of the flag 
            allow_missing.
    """

    def __init__(self, tokenizer, sim_measure_type, threshold,
                 allow_empty=True, allow_missing=False):
        # check if the sim_measure_type is valid                                
        validate_sim_measure_type(sim_measure_type)   
        sim_measure_type = sim_measure_type.upper()

        # check if the input tokenizer is valid
        validate_tokenizer_for_sim_measure(tokenizer, sim_measure_type)

        # check if the threshold is valid
        validate_threshold(threshold, sim_measure_type)

        self.tokenizer = tokenizer
        self.sim_measure_type = sim_measure_type
        self.threshold = threshold
        self.allow_empty = allow_empty

        super(self.__class__, self).__init__(allow_missing)

    def filter_pair(self, lstring, rstring):
        """Checks if the input strings get dropped by the size filter.

        Args:
            lstring,rstring (string): input strings

        Returns:
            A flag indicating whether the string pair is dropped (boolean).
        """

        # If one of the inputs is missing, then check the allow_missing flag.
        # If it is set to True, then pass the pair. Else drop the pair.
        if pd.isnull(lstring) or pd.isnull(rstring):
            return (not self.allow_missing)

        l_num_tokens = len(self.tokenizer.tokenize(lstring))
        r_num_tokens = len(self.tokenizer.tokenize(rstring))

        if l_num_tokens == 0 and r_num_tokens == 0:
            if self.sim_measure_type == 'OVERLAP':
                return True
            elif self.sim_measure_type == 'EDIT_DISTANCE':
                return False
            else:
                return (not self.allow_empty)

        size_lower_bound = get_size_lower_bound(l_num_tokens,
                                                self.sim_measure_type,
                                                self.threshold)
        size_upper_bound = get_size_upper_bound(l_num_tokens,
                                                self.sim_measure_type,
                                                self.threshold)

        if size_lower_bound <= r_num_tokens <= size_upper_bound:
            return False
        else:
            return True

    def filter_tables(self, ltable, rtable,
                      l_key_attr, r_key_attr,
                      l_filter_attr, r_filter_attr,
                      l_out_attrs=None, r_out_attrs=None,
                      l_out_prefix='l_', r_out_prefix='r_',
                      n_jobs=1, show_progress=True):
        """Finds candidate matching pairs of strings from the input tables using
        size filtering technique.

        Args:
            ltable (DataFrame): left input table.

            rtable (DataFrame): right input table.

            l_key_attr (string): key attribute in left table.

            r_key_attr (string): key attribute in right table.

            l_filter_attr (string): attribute in left table on which the filter 
                should be applied.                                              
                                                                                
            r_filter_attr (string): attribute in right table on which the filter
                should be applied.                                              
                                                                                
            l_out_attrs (list): list of attribute names from the left table to  
                be included in the output table (defaults to None).             
                                                                                
            r_out_attrs (list): list of attribute names from the right table to 
                be included in the output table (defaults to None).             
                                                                                
            l_out_prefix (string): prefix to be used for the attribute names    
                coming from the left table, in the output table                 
                (defaults to 'l\_').                                            
                                                                                
            r_out_prefix (string): prefix to be used for the attribute names    
                coming from the right table, in the output table                
                (defaults to 'r\_').                                            
                                                                                
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
            An output table containing tuple pairs that survive the filter      
            (DataFrame).
        """

        # check if the input tables are dataframes
        validate_input_table(ltable, 'left table')
        validate_input_table(rtable, 'right table')

        # check if the key attributes and filter attributes exist
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

        # check if the output attributes exist
        validate_output_attrs(l_out_attrs, ltable.columns,
                              r_out_attrs, rtable.columns)

        # check if the key attributes are unique and do not contain 
        # missing values
        validate_key_attr(l_key_attr, ltable, 'left table')
        validate_key_attr(r_key_attr, rtable, 'right table')

        # remove redundant attrs from output attrs.
        l_out_attrs = remove_redundant_attrs(l_out_attrs, l_key_attr)
        r_out_attrs = remove_redundant_attrs(r_out_attrs, r_key_attr)

        # get attributes to project.  
        l_proj_attrs = get_attrs_to_project(l_out_attrs,
                                            l_key_attr, l_filter_attr)
        r_proj_attrs = get_attrs_to_project(r_out_attrs,
                                            r_key_attr, r_filter_attr)

        # Do a projection on the input dataframes to keep only the required         
        # attributes. Then, remove rows with missing value in filter attribute  
        # from the input dataframes. Then, convert the resulting dataframes     
        # into ndarray.                                                         
        ltable_array = convert_dataframe_to_array(ltable, l_proj_attrs,         
                                                  l_filter_attr)                
        rtable_array = convert_dataframe_to_array(rtable, r_proj_attrs,         
                                                  r_filter_attr) 

        # computes the actual number of jobs to launch.
        n_jobs = min(get_num_processes_to_launch(n_jobs), len(rtable_array))

        if n_jobs <= 1:
            # if n_jobs is 1, do not use any parallel code.                     
            output_table = _filter_tables_split(
                                           ltable_array, rtable_array,
                                           l_proj_attrs, r_proj_attrs,                 
                                           l_key_attr, r_key_attr,
                                           l_filter_attr, r_filter_attr,
                                           self,
                                           l_out_attrs, r_out_attrs,
                                           l_out_prefix, r_out_prefix,
                                           show_progress)
        else:
            # if n_jobs is above 1, split the right table into n_jobs splits and    
            # filter each right table split with the whole of left table in a   
            # separate process.
            r_splits = split_table(rtable_array, n_jobs)
            results = Parallel(n_jobs=n_jobs)(delayed(_filter_tables_split)(
                                    ltable_array, r_splits[job_index],
                                    l_proj_attrs, r_proj_attrs,                 
                                    l_key_attr, r_key_attr,
                                    l_filter_attr, r_filter_attr,
                                    self,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    (show_progress and (job_index==n_jobs-1)))
                                for job_index in range(n_jobs))
            output_table = pd.concat(results)

        # If allow_missing flag is set, then compute all pairs with missing 
        # value in at least one of the filter attributes and then add it to the 
        # output obtained from applying the filter. 
        if self.allow_missing:
            missing_pairs = get_pairs_with_missing_value(
                                            ltable, rtable,
                                            l_key_attr, r_key_attr,
                                            l_filter_attr, r_filter_attr,
                                            l_out_attrs, r_out_attrs,
                                            l_out_prefix, r_out_prefix,
                                            False, show_progress)
            output_table = pd.concat([output_table, missing_pairs])

        # add an id column named '_id' to the output table.
        output_table.insert(0, '_id', range(0, len(output_table)))

        return output_table

    def find_candidates(self, probe_size, size_index):
        # probe size index to find candidates for the input probe_size.

        if not size_index.index:
            return set()

        size_lower_bound = get_size_lower_bound(probe_size,
                                                self.sim_measure_type,
                                                self.threshold)
        size_upper_bound = get_size_upper_bound(probe_size,
                                                self.sim_measure_type,
                                                self.threshold)

        if size_lower_bound > probe_size:
            return set()

        size_lower_bound = (size_index.min_length if
                            size_lower_bound < size_index.min_length else
                            size_lower_bound)

        size_upper_bound = (size_index.max_length if
                            size_upper_bound > size_index.max_length else
                            size_upper_bound)

        # find candidates from size index with size in the range 
        # [size_lower_bound, size_upper_bound]
        candidates = set()
        for cand_size in xrange(size_lower_bound, size_upper_bound + 1):
            for cand in size_index.probe(cand_size):
                candidates.add(cand)
        return candidates


def _filter_tables_split(ltable, rtable,
                         l_columns, r_columns,
                         l_key_attr, r_key_attr,
                         l_filter_attr, r_filter_attr,
                         size_filter, 
                         l_out_attrs, r_out_attrs,
                         l_out_prefix, r_out_prefix, show_progress):
    # find column indices of key attr, filter attr and output attrs in ltable
    l_key_attr_index = l_columns.index(l_key_attr)
    l_filter_attr_index = l_columns.index(l_filter_attr)
    l_out_attrs_indices = []
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, filter attr and output attrs in rtable
    r_key_attr_index = r_columns.index(r_key_attr)
    r_filter_attr_index = r_columns.index(r_filter_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # ignore allow_empty flag for OVERLAP and EDIT_DISTANCE measures.
    handle_empty = (size_filter.allow_empty and 
        size_filter.sim_measure_type not in ['OVERLAP', 'EDIT_DISTANCE'])

    # Build size index over ltable
    size_index = SizeIndex(ltable, l_filter_attr_index,
                           size_filter.tokenizer)
    # While building the index, we cache the record ids with empty set of 
    # tokens. This is needed to handle the allow_empty flag.
    cached_data = size_index.build(handle_empty)
    l_empty_records = cached_data['empty_records']

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)

    if show_progress:
        prog_bar = pyprind.ProgBar(len(rtable))

    for r_row in rtable:
        r_string = r_row[r_filter_attr_index]

        r_num_tokens = len(size_filter.tokenizer.tokenize(r_string))

        # If allow_empty flag is set and the current rtable record has empty set
        # of tokens in the filter attribute, then generate output pairs joining   
        # the current rtable record with those records in ltable with empty set 
        # of tokens in the filter attribute. These ltable record ids are cached 
        # in l_empty_records list which was constructed when building the size 
        # index.
        if handle_empty and r_num_tokens == 0:
            for l_id in l_empty_records:
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                                     ltable[l_id], r_row,
                                     l_key_attr_index, r_key_attr_index,
                                     l_out_attrs_indices,
                                     r_out_attrs_indices)
                else:
                    output_row = [ltable[l_id][l_key_attr_index],
                                  r_row[r_key_attr_index]]

                output_rows.append(output_row)
            continue
           
        # probe size index and find candidates
        candidates = size_filter.find_candidates(r_num_tokens, size_index)

        for cand in candidates:
            if has_output_attributes:
                output_row = get_output_row_from_tables(
                                     ltable[cand], r_row,
                                     l_key_attr_index, r_key_attr_index, 
                                     l_out_attrs_indices, r_out_attrs_indices)
            else:
                output_row = [ltable[cand][l_key_attr_index],
                              r_row[r_key_attr_index]]

            output_rows.append(output_row)

        if show_progress:
            prog_bar.update()

    output_header = get_output_header_from_tables(l_key_attr, r_key_attr,
                                                  l_out_attrs, r_out_attrs, 
                                                  l_out_prefix, r_out_prefix)

    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table
