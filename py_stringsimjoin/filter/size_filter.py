import pandas as pd
import pyprind

from py_stringsimjoin.filter.filter import Filter
from py_stringsimjoin.filter.filter_utils import get_size_lower_bound
from py_stringsimjoin.filter.filter_utils import get_size_upper_bound
from py_stringsimjoin.index.size_index import SizeIndex
from py_stringsimjoin.utils.helper_functions import build_dict_from_table
from py_stringsimjoin.utils.helper_functions import \
                                                 find_output_attribute_indices
from py_stringsimjoin.utils.helper_functions import \
                                                 get_output_header_from_tables
from py_stringsimjoin.utils.helper_functions import get_output_row_from_tables
from py_stringsimjoin.utils.tokenizers import tokenize


class SizeFilter(Filter):
    """Size filter class.

    Attributes:
        tokenizer: Tokenizer object.
        sim_measure_type: String, similarity measure type.
        threshold: float, similarity threshold to be used by the filter. 
    """
    def __init__(self, tokenizer, sim_measure_type, threshold):
        self.tokenizer = tokenizer
        self.sim_measure_type = sim_measure_type
        self.threshold = threshold
        super(self.__class__, self).__init__()

    def filter_pair(self, lstring, rstring):
        """Filter two strings with size filter.

        Args:
        lstring, rstring : input strings

        Returns:
        result : boolean, True if the tuple pair is dropped.
        """
        # check for empty string
        if (not lstring) or (not rstring):
            return True

        l_num_tokens = len(tokenize(lstring, self.tokenizer,
                                    self.sim_measure_type))
        r_num_tokens = len(tokenize(rstring, self.tokenizer,
                                    self.sim_measure_type))

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
                      l_out_prefix='l_', r_out_prefix='r_'):
        """Filter tables with size filter.

        Args:
        ltable, rtable : Pandas data frame
        l_key_attr, r_key_attr : String, key attribute from ltable and rtable
        l_filter_attr, r_filter_attr : String, filter attribute from ltable and rtable
        l_out_attrs, r_out_attrs : list of attribtues to be included in the output table from ltable and rtable
        l_out_prefix, r_out_prefix : String, prefix to be used in the attribute names of the output table 

        Returns:
        result : Pandas data frame
        """
        # find column indices of key attr, filter attr and
        # output attrs in ltable
        l_columns = list(ltable.columns.values)
        l_key_attr_index = l_columns.index(l_key_attr)
        l_filter_attr_index = l_columns.index(l_filter_attr)
        l_out_attrs_indices = []
        l_out_attrs_indices = find_output_attribute_indices(l_columns,
                                                            l_out_attrs)

        # find column indices of key attr, filter attr and
        # output attrs in rtable
        r_columns = list(rtable.columns.values)
        r_key_attr_index = r_columns.index(r_key_attr)
        r_filter_attr_index = r_columns.index(r_filter_attr)
        r_out_attrs_indices = find_output_attribute_indices(r_columns,
                                                            r_out_attrs)

        # build a dictionary on ltable
        ltable_dict = build_dict_from_table(ltable, l_key_attr_index,
                                            l_filter_attr_index)

        # build a dictionary on rtable
        rtable_dict = build_dict_from_table(rtable, r_key_attr_index,
                                            r_filter_attr_index)

        # Build size index over ltable
        size_index = SizeIndex(ltable_dict.values(),
                               l_key_attr_index, l_filter_attr_index,
                               self.tokenizer)
        size_index.build()

        output_rows = []
        has_output_attributes = (l_out_attrs is not None or
                                 r_out_attrs is not None)
        prog_bar = pyprind.ProgBar(len(rtable.index))

        for r_row in rtable_dict.values():
            r_id = r_row[r_key_attr_index]
            r_string = str(r_row[r_filter_attr_index])
            # check for empty string
            if not r_string:
                continue
            r_num_tokens = len(tokenize(r_string, self.tokenizer,
                                        self.sim_measure_type))
           
            size_lower_bound = get_size_lower_bound(r_num_tokens,
                                                    self.sim_measure_type,
                                                    self.threshold)
            size_upper_bound = get_size_upper_bound(r_num_tokens,
                                                    self.sim_measure_type,
                                                    self.threshold)

            size_lower_bound = (size_index.min_length if
                                size_lower_bound < size_index.min_length else 
                                size_lower_bound) 

            size_upper_bound = (size_index.max_length if
                                size_upper_bound > size_index.max_length else 
                                size_upper_bound) 

            # probe size index and find candidates
            candidates = set()
            for cand_size in xrange(size_lower_bound, size_upper_bound + 1):
                for cand in size_index.probe(cand_size):
                    candidates.add(cand)

            for cand in candidates:
                if has_output_attributes:
                    output_row = get_output_row_from_tables(
                                     ltable_dict[cand], r_row,
                                     cand, r_id, 
                                     l_out_attrs_indices, r_out_attrs_indices)
                    output_rows.append(output_row)
                else:
                    output_rows.append([cand, r_id])

            prog_bar.update()

        output_header = get_output_header_from_tables(
                            l_key_attr, r_key_attr,
                            l_out_attrs, r_out_attrs, 
                            l_out_prefix, r_out_prefix)

        output_table = pd.DataFrame(output_rows, columns=output_header)
        output_table.insert(0, '_id', range(0, len(output_table)))
        return output_table
