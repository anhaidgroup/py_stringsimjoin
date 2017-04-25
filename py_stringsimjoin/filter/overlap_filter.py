# Overlap Filter

from joblib import delayed, Parallel
from six import iteritems
import pandas as pd
import pyprind

from py_stringsimjoin.filter.filter import Filter
from py_stringsimjoin.index.inverted_index import InvertedIndex
from py_stringsimjoin.utils.generic_helper import convert_dataframe_to_array, \
    find_output_attribute_indices, get_attrs_to_project, \
    get_num_processes_to_launch, get_output_header_from_tables, \
    get_output_row_from_tables, remove_redundant_attrs, split_table, COMP_OP_MAP, get_temp_filenames, add_id_to_file, \
    remove_files, merge_outputs_and_add_id
from py_stringsimjoin.utils.missing_value_handler import \
    get_pairs_with_missing_value
from py_stringsimjoin.utils.simfunctions import overlap
from py_stringsimjoin.utils.validation import validate_attr, \
    validate_attr_type, validate_comp_op_for_sim_measure, validate_key_attr, \
    validate_input_table, validate_threshold, validate_tokenizer, \
    validate_output_attrs


class OverlapFilter(Filter):
    """Finds candidate matching pairs of strings using overlap filtering 
    technique.

    A string pair is output by overlap filter only if the number of common 
    tokens in the strings satisfy the condition on overlap size threshold. For 
    example, if the comparison operator is '>=', a string pair is output if the 
    number of common tokens is greater than or equal to the overlap size 
    threshold, as specified by "overlap_size".

    Args:
        tokenizer (Tokenizer): tokenizer to be used.
        overlap_size (int): overlap threshold to be used by the filter.
        comp_op (string): comparison operator. Supported values are '>=', '>' 
            and '=' (defaults to '>=').  
        allow_missing (boolean): A flag to indicate whether pairs containing 
            missing value should survive the filter (defaults to False). 
        file_name (string): Location where output will be stored.
        mem_threshold (int): Memory Threshold which is the limit of intermediate data that can be written to memory.
        append_to_file (boolean): A flag to indicate whether appending is enabled or disabled
    Attributes:
        tokenizer (Tokenizer): An attribute to store the tokenizer.
        overlap_size (int): An attribute to store the overlap threshold value.
        comp_op (string): An attribute to store the comparison operator.
        allow_missing (boolean): An attribute to store the value of the flag 
            allow_missing.
        file_name (string): An attribute to store the output location.
        mem_threshold (int): An attribute to specify the memory threshold.
        append_to_file (boolean): An attribute to store the value of flag append_to_file.
    """

    def __init__(self, tokenizer, overlap_size=1, comp_op='>=',
                 allow_missing=False,file_name=None, mem_threshold=1e9, append_to_file=False):
        # check if the input tokenizer is valid
        validate_tokenizer(tokenizer)

        # check if the overlap size is valid
        validate_threshold(overlap_size, 'OVERLAP')

        # check if the comparison operator is valid
        validate_comp_op_for_sim_measure(comp_op, 'OVERLAP')

        self.tokenizer = tokenizer
        self.overlap_size = overlap_size
        self.comp_op = comp_op
        self.file_name = file_name
        self.mem_threshold = mem_threshold
        self.append_to_file = append_to_file
        
        super(self.__class__, self).__init__(allow_missing)

    def filter_pair(self, lstring, rstring):
        """Checks if the input strings get dropped by the overlap filter.

        Args:
            lstring,rstring (string): input strings

        Returns:
            A flag indicating whether the string pair is dropped (boolean).
        """

        # If one of the inputs is missing, then check the allow_missing flag.
        # If it is set to True, then pass the pair. Else drop the pair.
        if pd.isnull(lstring) or pd.isnull(rstring):
            return (not self.allow_missing)

        # check for empty string
        if (not lstring) or (not rstring):
            return True

        # tokenize input strings 
        ltokens = self.tokenizer.tokenize(lstring)
        rtokens = self.tokenizer.tokenize(rstring)
 
        num_overlap = overlap(ltokens, rtokens) 

        if COMP_OP_MAP[self.comp_op](num_overlap, self.overlap_size):
            return False
        else:
            return True

    def filter_tables(self, ltable, rtable,
                      l_key_attr, r_key_attr,
                      l_filter_attr, r_filter_attr,
                      l_out_attrs=None, r_out_attrs=None,
                      l_out_prefix='l_', r_out_prefix='r_',
                      out_sim_score=False, n_jobs=1, show_progress=True,file_name=None, mem_threshold=1e9, append_to_file=False):
        """Finds candidate matching pairs of strings from the input tables using
        overlap filtering technique.

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
            
            file_name (string): Location where output will be stored.
            
            mem_threshold (int): Memory Threshold which is the limit of 
                intermediate data that can be written to memory.
            
            append_to_file (boolean): A flag to indicate whether appending
                is enabled or disabled.

        Returns:
            A boolean value to indicate completion of operation.
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
                      'attribute', 'left table')
        validate_attr(r_filter_attr, rtable.columns,
                      'attribute', 'right table')

        # check if the filter attributes are not of numeric type                      
        validate_attr_type(l_filter_attr, ltable[l_filter_attr].dtype,                  
                           'attribute', 'left table')                          
        validate_attr_type(r_filter_attr, rtable[r_filter_attr].dtype,                  
                           'attribute', 'right table')

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

        # Call the respective join function depending on whether join is to
        # be done in-memory or out of memory
        output = None
        if file_name == None:
            #join in memory. Tuple pair functionality will not be used in this case.
            from py_stringmatching.tokenizer import tokenizer
            from symbol import comp_op
            output = self._overlap_join_in_mem(ltable, rtable, ltable_array, rtable_array, l_proj_attrs, r_proj_attrs,
                                               l_key_attr, r_key_attr,
                                               l_filter_attr, r_filter_attr,
                                               l_out_attrs, r_out_attrs,
                                               l_out_prefix, r_out_prefix,
                                               out_sim_score, n_jobs, show_progress)
        else:
            #join out of memory and using Tuple pair functionality.
            output = self._overlap_join_ooc_mem(ltable, rtable, ltable_array, rtable_array, l_proj_attrs, r_proj_attrs,
                                                l_key_attr, r_key_attr,
                                                l_filter_attr, r_filter_attr,
                                                l_out_attrs, r_out_attrs,
                                                l_out_prefix, r_out_prefix,
                                                out_sim_score, n_jobs, show_progress,
                                                file_name, mem_threshold, append_to_file)

        return output
                                               
    #Args are same as used in above functions
    def _overlap_join_ooc_mem(self, ltable, rtable,
                                           ltable_array, rtable_array,
                                           l_proj_attrs, r_proj_attrs,
                                           l_key_attr, r_key_attr,
                                           l_filter_attr, r_filter_attr,
                                           l_out_attrs, r_out_attrs,
                                           l_out_prefix, r_out_prefix,
                                           out_sim_score, n_jobs, show_progress,
                                           file_name, mem_threshold, append_to_file):

        # computes the actual number of jobs to launch.
        n_jobs = min(get_num_processes_to_launch(n_jobs), len(rtable_array))
        if n_jobs <= 1:
            # if n_jobs is 1, do not use any parallel code.
            # scratch_dir = tempfile.mkdtemp()
            # Intermediate file locations to buffer the intermediate parallel outputs.
            from tempfile import mkdtemp
            scratch_dir = mkdtemp()
            _intermediate_file_names = get_temp_filenames(1, scratch_dir)
            _intermediate_file_name = _intermediate_file_names[0]

            output = _filter_tables_split(
                                           ltable_array, rtable_array,
                                           l_proj_attrs, r_proj_attrs,
                                           l_key_attr, r_key_attr,
                                           l_filter_attr, r_filter_attr,
                                           self,
                                           l_out_attrs, r_out_attrs,
                                           l_out_prefix, r_out_prefix,
                                           out_sim_score, show_progress,
                                           _intermediate_file_name,
                                           mem_threshold, append_to_file)
            if self.allow_missing:
                missing_pairs = get_pairs_with_missing_value(
                    ltable, rtable,
                    l_key_attr, r_key_attr,
                    l_filter_attr, r_filter_attr,
                    l_out_attrs, r_out_attrs,
                    l_out_prefix, r_out_prefix,
                    out_sim_score, show_progress,
                    _intermediate_file_name, mem_threshold, append_to_file=True)

            add_id_to_file(_intermediate_file_name, file_name, mem_threshold=mem_threshold)
            remove_files([_intermediate_file_name])



        else:
            # if n_jobs is above 1, split the right table into n_jobs splits and
            # join each right table split with the whole of left table in a separate
            # process.
            r_splits = split_table(rtable_array, n_jobs)
            import tempfile
            scratch_dir = tempfile.mkdtemp()
            _intermediate_file_names = get_temp_filenames(n_jobs, scratch_dir)

            import math
            results = Parallel(n_jobs=n_jobs)(delayed(_filter_tables_split)(
                                    ltable_array, r_splits[job_index],
                                    l_proj_attrs, r_proj_attrs,
                                    l_key_attr, r_key_attr,
                                    l_filter_attr, r_filter_attr,
                                    self,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score,
                                    (show_progress and (job_index==n_jobs-1)),
                                    _intermediate_file_names[job_index],
                                    math.ceil(mem_threshold / 4), append_to_file=False)
                                              for job_index in range(n_jobs))

            if self.allow_missing:
                miss_file_names = get_temp_filenames(1, scratch_dir)
                miss_file_name = miss_file_names[0]
                get_pairs_with_missing_value(
                    ltable, rtable,
                    l_key_attr, r_key_attr,
                    l_filter_attr, r_filter_attr,
                    l_out_attrs, r_out_attrs,
                    l_out_prefix, r_out_prefix,
                    out_sim_score, show_progress,
                    miss_file_name, mem_threshold, append_to_file=True)

                _intermediate_file_names.append(miss_file_name)

            status = merge_outputs_and_add_id(_intermediate_file_names, file_name,
                                              mem_threshold=mem_threshold)

            remove_files(_intermediate_file_names)

        # Do not return the dataframes here as size can be massive. We just need to return boolean value back to caller.
        return True

    def _overlap_join_in_mem(              self, ltable, rtable,
                                           ltable_array, rtable_array,
                                           l_proj_attrs, r_proj_attrs,
                                           l_key_attr, r_key_attr,
                                           l_filter_attr, r_filter_attr,

                                           l_out_attrs, r_out_attrs,
                                           l_out_prefix, r_out_prefix,
                                           out_sim_score, n_jobs, show_progress):
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
                                           out_sim_score, show_progress)
        else:
            # if n_jobs is above 1, split the right table into n_jobs splits and
            # join each right table split with the whole of left table in a separate
            # process.
            r_splits = split_table(rtable_array, n_jobs)
            results = Parallel(n_jobs=n_jobs)(delayed(_filter_tables_split)(
                                    ltable_array, r_splits[job_index],
                                    l_proj_attrs, r_proj_attrs,
                                    l_key_attr, r_key_attr,
                                    l_filter_attr, r_filter_attr,
                                    self,
                                    l_out_attrs, r_out_attrs,
                                    l_out_prefix, r_out_prefix,
                                    out_sim_score,
                                    (show_progress and (job_index==n_jobs-1)))
                                for job_index in range(n_jobs))
            output_table = pd.concat(results)

        # If allow_missing flag is set, then compute all pairs with missing value in
        # at least one of the join attributes and then add it to the output
        # obtained from the join.
        if self.allow_missing:
            missing_pairs = get_pairs_with_missing_value(
                ltable, rtable,
                l_key_attr, r_key_attr,
                l_filter_attr, r_filter_attr,
                l_out_attrs, r_out_attrs,
                l_out_prefix, r_out_prefix,
                out_sim_score, show_progress)
            output_table = pd.concat([output_table, missing_pairs])

        # add an id column named '_id' to the output table.
        output_table.insert(0, '_id', range(0, len(output_table)))
        output_table.reset_index(drop=True, inplace=True)
        
        return output_table

    def find_candidates(self, probe_tokens, inverted_index):
        candidate_overlap = {}

        if not inverted_index.index:
            return candidate_overlap

        for token in probe_tokens:
            for cand in inverted_index.probe(token):
                candidate_overlap[cand] = candidate_overlap.get(cand, 0) + 1
        return candidate_overlap


def _filter_tables_split(ltable, rtable,
                         l_columns, r_columns,
                         l_key_attr, r_key_attr,
                         l_filter_attr, r_filter_attr,
                         overlap_filter,
                         l_out_attrs, r_out_attrs,
                         l_out_prefix, r_out_prefix,
                         out_sim_score, show_progress,
                         file_name=None, mem_threshold=1e9,
                         append_to_file=False):
    # Find column indices of key attr, filter attr and output attrs in ltable
    l_key_attr_index = l_columns.index(l_key_attr)
    l_filter_attr_index = l_columns.index(l_filter_attr)
    l_out_attrs_indices = []
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # Find column indices of key attr, filter attr and output attrs in rtable
    r_key_attr_index = r_columns.index(r_key_attr)
    r_filter_attr_index = r_columns.index(r_filter_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)

    # Build inverted index over ltable
    inverted_index = InvertedIndex(ltable, l_filter_attr_index,
                                   overlap_filter.tokenizer)
    inverted_index.build(False)

    comp_fn = COMP_OP_MAP[overlap_filter.comp_op]

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)

    if show_progress:
        prog_bar = pyprind.ProgBar(len(rtable))
    
    # Need to insert output header row to each parallel intermediate file
    # before filling them with any actual tuple rows. So forming output_header here.
    output_header = get_output_header_from_tables(
        l_key_attr, r_key_attr,
        l_out_attrs, r_out_attrs,
        l_out_prefix, r_out_prefix)

    if out_sim_score:
        output_header.append("_sim_score")

    from py_stringsimjoin.utils.tuple_pair_chest import TuplePairChest
    chest = TuplePairChest(file_name=file_name, mem_size=mem_threshold,
                           header=output_header,
                           append_to_file=append_to_file)

    # This is where header row is inserted in each chest at the beginning.
    chest.preprocess()

    for r_row in rtable:
        r_string = r_row[r_filter_attr_index]
        r_filter_attr_tokens = overlap_filter.tokenizer.tokenize(r_string)

        # probe inverted index and find overlap of candidates          
        candidate_overlap = overlap_filter.find_candidates(
                                r_filter_attr_tokens, inverted_index)

        for cand, overlap in iteritems(candidate_overlap):
            if comp_fn(overlap, overlap_filter.overlap_size):
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                                     ltable[cand], r_row,
                                     l_key_attr_index, r_key_attr_index,
                                     l_out_attrs_indices, r_out_attrs_indices)
                else:
                    output_row = [ltable[cand][l_key_attr_index],
                                  r_row[r_key_attr_index]]

                if out_sim_score:
                    output_row.append(overlap)
                chest.append(output_row)
 
        if show_progress:
            prog_bar.update()




    # output can be boolean or dataframe type depending on whether
    # file_name was specified or not. If file_name was specified,
    # it is inferred as memory-constrained operation hence only boolean returned.
    output = chest.postprocess()
    return output
