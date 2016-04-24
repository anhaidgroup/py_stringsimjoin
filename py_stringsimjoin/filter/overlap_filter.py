import pandas as pd
import pyprind

from py_stringsimjoin.filter.filter import Filter
from py_stringsimjoin.index.inverted_index import InvertedIndex
from py_stringsimjoin.utils.helper_functions import \
                                                 get_output_header_from_tables
from py_stringsimjoin.utils.helper_functions import get_output_row_from_tables


class OverlapFilter(Filter):
    """Overlap filter class.

    Attributes:
        tokenizer: Tokenizer function, which is used to tokenize input string.
        overlap_size: overlap threshold for the filter.
    """
    def __init__(self, tokenizer, overlap_size=1):
        self.tokenizer = tokenizer
        self.overlap_size = overlap_size
        super(self.__class__, self).__init__()

    def filter_pair(self, lstring, rstring):
        """Filter two strings with overlap filter.

        Args:
        lstring, rstring : input strings

        Returns:
        result : boolean, True if the tuple pair is dropped.
        """
        ltokens = self.tokenizer(lstring)
        rtokens = self.tokenizer(rstring)
 
        num_overlap = len(set(ltokens).intersection(set(rtokens))) 

        if num_overlap < self.overlap_size:
            return True
        else:
            return False

    def filter_tables(self, ltable, rtable,
                      l_id_attr, r_id_attr,
                      l_filter_attr, r_filter_attr,
                      l_out_attrs=None, r_out_attrs=None,
                      l_out_prefix='l_', r_out_prefix='r_'):
        """Filter tables with overlap filter.

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

        # build a dictionary on rtable
        rtable_dict = {}
        for r_row in rtable.itertuples(index=False):
            rtable_dict[r_row[r_id_attr_index]] = r_row

        # Build inverted index over ltable
        inverted_index = InvertedIndex(ltable_dict.values(),
                                       l_id_attr_index, l_filter_attr_index,
                                       self.tokenizer)
        inverted_index.build()

        output_rows = []
        has_output_attributes = (l_out_attrs is not None or
                                 r_out_attrs is not None)
        prog_bar = pyprind.ProgBar(len(rtable.index))
        candset_id = 1

        for r_row in rtable_dict.values():
            r_id = r_row[r_id_attr_index]
            r_filter_attr_tokens = set(self.tokenizer(str(
                                       r_row[r_filter_attr_index])))

            # probe inverted index and find overlap of candidates          
            candidate_overlap = {}
            for token in r_filter_attr_tokens:
                for cand in inverted_index.probe(token):
                    candidate_overlap[cand] = candidate_overlap.get(cand, 0) + 1

            for cand, overlap in candidate_overlap.iteritems():
                if overlap >= self.overlap_size:
                    if has_output_attributes:
                        output_row = get_output_row_from_tables(
                                         candset_id,
                                         ltable_dict[cand], r_row,
                                         cand, r_id, 
                                         l_out_attrs_indices,
                                         r_out_attrs_indices)
                        output_rows.append(output_row)
                    else:
                        output_rows.append([candset_id, cand, r_id])

                    candset_id += 1
            prog_bar.update()

        output_header = get_output_header_from_tables(
                            '_id',
                            l_id_attr, r_id_attr,
                            l_out_attrs, r_out_attrs, 
                            l_out_prefix, r_out_prefix)
        output_table = pd.DataFrame(output_rows, columns=output_header)
        return output_table
