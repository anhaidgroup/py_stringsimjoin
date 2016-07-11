# Prefix Filter

from joblib import delayed, Parallel
import pandas as pd
import pyprind

from py_stringsimjoin.filter.filter import Filter
from py_stringsimjoin.filter.filter_utils import get_prefix_length
from py_stringsimjoin.index.prefix_index import PrefixIndex
from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_array, \
    find_output_attribute_indices, get_attrs_to_project, \
    get_num_processes_to_launch, get_output_header_from_tables, \
    get_output_row_from_tables, remove_redundant_attrs, split_table
from py_stringsimjoin.utils.missing_value_handler import \
    get_pairs_with_missing_value
from py_stringsimjoin.utils.token_ordering import gen_token_ordering_for_lists, \
    gen_token_ordering_for_tables, order_using_token_ordering
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_key_attr, validate_input_table, \
    validate_threshold, validate_tokenizer_for_sim_measure, \
    validate_output_attrs, validate_sim_measure_type


class PrefixFilter(Filter):
    """Finds candidate matching pairs of strings using prefix filtering 
    technique.

    For similarity measures such as cosine, Dice, Jaccard and overlap, the 
    filter finds candidate string pairs that may have similarity score greater 
    than or equal to the input threshold, as specified in "threshold". For 
    distance measures such as edit distance, the filter finds candidate string 
    pairs that may have distance score less than or equal to the threshold.

    To know more about prefix filtering, refer to the `string matching chapter 
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
        """Checks if the input strings get dropped by the prefix filter.

        Args:
            lstring,rstring (string): input strings

        Returns:
            A flag indicating whether the string pair is dropped (boolean).
        """

        # If one of the inputs is missing, then check the allow_missing flag.
        # If it is set to True, then pass the pair. Else drop the pair.
        if pd.isnull(lstring) or pd.isnull(rstring):
            return (not self.allow_missing)

        # tokenize input strings 
        ltokens = self.tokenizer.tokenize(lstring)
        rtokens = self.tokenizer.tokenize(rstring)

        l_num_tokens = len(ltokens)
        r_num_tokens = len(rtokens)

        if l_num_tokens == 0 and r_num_tokens == 0:
            if self.sim_measure_type == 'OVERLAP':
                return True
            elif self.sim_measure_type == 'EDIT_DISTANCE':
                return False
            else:
                return (not self.allow_empty)
        
        token_ordering = gen_token_ordering_for_lists([ltokens, rtokens])
        ordered_ltokens = order_using_token_ordering(ltokens, token_ordering)
        ordered_rtokens = order_using_token_ordering(rtokens, token_ordering)

        l_prefix_length = get_prefix_length(l_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold,
                                            self.tokenizer) 
        r_prefix_length = get_prefix_length(r_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold,
                                            self.tokenizer)

        if l_prefix_length <= 0 or r_prefix_length <= 0:
            return True

        prefix_overlap = set(ordered_ltokens[0:l_prefix_length]).intersection(
                         set(ordered_rtokens[0:r_prefix_length]))

        if len(prefix_overlap) > 0:
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
        prefix filtering technique.

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
                                                                                
            out_sim_score (boolean): flag to indicate whether the overlap score 
                should be included in the output table (defaults to True).      
                Setting this flag to True will add a column named '_sim_score'  
                in the output table. This column will contain the overlap scores
                for the tuple pairs in the output.                              

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

    def find_candidates(self, probe_tokens, prefix_index):
        # probe prefix index to find candidates for the input probe tokens.

        if not prefix_index.index:
            return set()

        probe_num_tokens = len(probe_tokens)
        probe_prefix_length = get_prefix_length(probe_num_tokens,
                                                self.sim_measure_type,
                                                self.threshold,
                                                self.tokenizer)

        candidates = set()
        for token in probe_tokens[0:probe_prefix_length]:
            candidates.update(prefix_index.probe(token))
        return candidates


def _filter_tables_split(ltable, rtable,
                         l_columns, r_columns,
                         l_key_attr, r_key_attr,
                         l_filter_attr, r_filter_attr,
                         prefix_filter,
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

    # generate token ordering using tokens in l_filter_attr and r_filter_attr
    token_ordering = gen_token_ordering_for_tables(
                                 [ltable, rtable],
                                 [l_filter_attr_index, r_filter_attr_index],
                                 prefix_filter.tokenizer,
                                 prefix_filter.sim_measure_type)

    # ignore allow_empty flag for OVERLAP and EDIT_DISTANCE measures.           
    handle_empty = (prefix_filter.allow_empty and
        prefix_filter.sim_measure_type not in ['OVERLAP', 'EDIT_DISTANCE'])

    # Build prefix index on l_filter_attr
    prefix_index = PrefixIndex(ltable, l_filter_attr_index, 
                       prefix_filter.tokenizer, prefix_filter.sim_measure_type,
                       prefix_filter.threshold, token_ordering)
    # While building the index, we cache the record ids with empty set of 
    # tokens. This is needed to handle the allow_empty flag.
    cached_data = prefix_index.build(handle_empty)
    l_empty_records = cached_data['empty_records']

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)

    if show_progress:
        prog_bar = pyprind.ProgBar(len(rtable))

    for r_row in rtable:
        r_string = r_row[r_filter_attr_index]

        r_filter_attr_tokens = prefix_filter.tokenizer.tokenize(r_string)
        r_ordered_tokens = order_using_token_ordering(r_filter_attr_tokens,
                                                      token_ordering)

        # If allow_empty flag is set and the current rtable record has empty set
        # of tokens in the filter attribute, then generate output pairs joining   
        # the current rtable record with those records in ltable with empty set 
        # of tokens in the filter attribute. These ltable record ids are cached 
        # in l_empty_records list which was constructed when building the prefix  
        # index. 
        if handle_empty and len(r_ordered_tokens) == 0:
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
           
        # probe prefix index and find candidates
        candidates = prefix_filter.find_candidates(r_ordered_tokens,
                                                   prefix_index)

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

    # generate a dataframe from the list of output rows
    output_table = pd.DataFrame(output_rows, columns=output_header)
    return output_table
