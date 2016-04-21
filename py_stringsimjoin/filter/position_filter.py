import pandas as pd
import pyprind

from py_stringsimjoin.filter.filter import Filter
from py_stringsimjoin.filter.filter_utils import get_overlap_threshold
from py_stringsimjoin.filter.filter_utils import get_prefix_length
from py_stringsimjoin.filter.filter_utils import get_size_lower_bound
from py_stringsimjoin.filter.filter_utils import get_size_upper_bound
from py_stringsimjoin.index.position_index import PositionIndex
from py_stringsimjoin.utils.helper_functions import \
                                                 get_output_header_from_tables
from py_stringsimjoin.utils.helper_functions import get_output_row_from_tables
from py_stringsimjoin.utils.token_ordering import gen_token_ordering
from py_stringsimjoin.utils.token_ordering import order_using_token_ordering


class PositionFilter(Filter):
    """Position filter class.

    Attributes:
        tokenizer: Tokenizer function, which is used to tokenize input string.
        sim_measure_type: String, similarity measure type.
        threshold: float, similarity threshold to be used by the filter.
    """
    def __init__(self, tokenizer, sim_measure_type, threshold):
        self.tokenizer = tokenizer
        self.sim_measure_type = sim_measure_type
        self.threshold = threshold
        super(self.__class__, self).__init__()

    def filter_pair(self, lstring, rstring):
        """Filter two strings with position filter.

        Args:
        lstring, rstring : input strings

        Returns:
        result : boolean, True if the tuple pair is dropped.
        """
        ltokens = list(set(self.tokenizer(lstring)))
        rtokens = list(set(self.tokenizer(rstring)))

        l_num_tokens = len(ltokens)
        r_num_tokens = len(rtokens)

        l_prefix_length = get_prefix_length(l_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold) 
        r_prefix_length = get_prefix_length(r_num_tokens,
                                            self.sim_measure_type,
                                            self.threshold)
 
        l_prefix_dict = {}
        l_pos = 0
        for token in ltokens[0:l_prefix_length]:
            l_prefix_dict[token] = l_pos

        overlap_threshold = get_overlap_threshold(l_num_tokens, r_num_tokens,
                                                  self.sim_measure_type,
                                                  self.threshold)
        current_overlap = 0
        r_pos = 0 
        for token in rtokens[0:r_prefix_length]:
            l_pos = l_prefix_dict.get(token)
            if l_pos is not None:
                overlap_upper_bound = 1 + min(l_num_tokens - l_pos - 1,
                                              r_num_tokens - r_pos - 1)
                if (current_overlap + overlap_upper_bound) < overlap_threshold:
                    return True
                current_overlap += 1
            r_pos += 1

        if current_overlap > 0:
            return False
        return True
        
    def filter_tables(self, ltable, rtable,
                      l_id_attr, r_id_attr,
                      l_filter_attr, r_filter_attr,
                      l_out_attrs=None, r_out_attrs=None,
                      l_out_prefix='l_', r_out_prefix='r_'):
        """Filter tables with position filter.

        Args:
        ltable, rtable : Pandas data frame
        l_id_attr, r_id_attr : String, id attribute from ltable and rtable
        l_filter_attr, r_filter_attr : String, filter attribute from ltable and rtable
        l_out_attrs, r_out_attrs : list of attribtues to be included in the output table from ltable and rtable
        l_out_prefix, r_out_prefix : String, prefix to be used in the attribute names of the output table 

        Returns:
        result : Pandas data frame
        """
        # find column indices of id attr, filter attr and output attrs in ltable
        l_columns = list(ltable.columns.values)
        l_id_attr_index = l_columns.index(l_id_attr)
        l_filter_attr_index = l_columns.index(l_filter_attr)
        l_out_attrs_indices = []
        if l_out_attrs is not None:
            for attr in l_out_attrs:
                l_out_attrs_indices.append(l_columns.index(attr))        

        # find column indices of id attr, filter attr and output attrs in rtable
        r_columns = list(rtable.columns.values)
        r_id_attr_index = r_columns.index(r_id_attr)
        r_filter_attr_index = r_columns.index(r_filter_attr)
        r_out_attrs_indices = []
        if r_out_attrs:
            for attr in r_out_attrs:
                r_out_attrs_indices.append(r_columns.index(attr))
        
        # build a dictionary on ltable
        ltable_dict = {}
        for l_row in ltable.itertuples(index=False):
            ltable_dict[l_row[l_id_attr_index]] = l_row

        # generate token ordering using tokens in l_filter_attr
        token_ordering = gen_token_ordering(ltable_dict.values(),
                                            l_filter_attr_index,
                                            self.tokenizer) 
        # build a dictionary on rtable
        rtable_dict = {}
        for r_row in rtable.itertuples(index=False):
            rtable_dict[r_row[r_id_attr_index]] = r_row

        # Build position index on l_filter_attr
        position_index = PositionIndex(ltable_dict.values(),
                                       l_id_attr_index, l_filter_attr_index,
                                       self.tokenizer, self.sim_measure_type,
                                       self.threshold, token_ordering)
        position_index.build()

        output_rows = []
        has_output_attributes = (l_out_attrs is not None or
                                 r_out_attrs is not None)
        prog_bar = pyprind.ProgBar(len(rtable.index))

        for r_row in rtable_dict.values():
            r_id = r_row[r_id_attr_index]
            r_filter_attr_tokens = set(self.tokenizer(str(
                                       r_row[r_filter_attr_index])))
            r_ordered_tokens = order_using_token_ordering(r_filter_attr_tokens,
                                                          token_ordering)
            r_num_tokens = len(r_ordered_tokens)
            size_lower_bound = get_size_lower_bound(r_num_tokens,
                                                    self.sim_measure_type,
                                                    self.threshold)
            size_upper_bound = get_size_upper_bound(r_num_tokens,
                                                    self.sim_measure_type,
                                                    self.threshold)

            overlap_threshold_cache = {}
            for size in xrange(size_lower_bound, size_upper_bound + 1):
                overlap_threshold_cache[size] = get_overlap_threshold(
                                                    size, r_num_tokens,
                                                    self.sim_measure_type,
                                                    self.threshold)

            r_prefix_length = get_prefix_length(r_num_tokens,
                                                self.sim_measure_type,
                                                self.threshold)

            # probe position index and find candidates
            candidate_overlap = {}
            r_pos = 0
            for token in r_ordered_tokens[0:r_prefix_length]:
                for (cand, cand_pos)  in position_index.probe(token):
                    cand_num_tokens = position_index.get_size(cand)
                    if size_lower_bound <= cand_num_tokens <= size_upper_bound:
                        overlap_upper_bound = 1 + min(r_num_tokens - r_pos - 1,
                                                cand_num_tokens - cand_pos - 1)
                        current_overlap = candidate_overlap.get(cand, 0)
                        if (current_overlap + overlap_upper_bound >=
                               overlap_threshold_cache[cand_num_tokens]):
                            candidate_overlap[cand] = current_overlap + 1
                        else:
                            candidate_overlap[cand] = 0
                r_pos += 1
                            
            for cand, overlap in candidate_overlap.iteritems():
                if overlap > 0:
                    if has_output_attributes:
                        output_row = get_output_row_from_tables(
                                         ltable_dict[cand], r_row,
                                         cand, r_id, 
                                         l_out_attrs_indices,
                                         r_out_attrs_indices)
                        output_rows.append(output_row)
                    else:
                        output_rows.append([cand, r_id])

            prog_bar.update()

        output_header = get_output_header_from_tables(l_id_attr, r_id_attr,
                            l_out_attrs, r_out_attrs, 
                            l_out_prefix, r_out_prefix)
        # generate a dataframe from the list of output rows
        output_table = pd.DataFrame(output_rows, columns=output_header)
        return output_table
