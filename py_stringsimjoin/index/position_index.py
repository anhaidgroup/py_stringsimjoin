from sys import maxsize

from py_stringsimjoin.filter.filter_utils import get_prefix_length
from py_stringsimjoin.index.index import Index
from py_stringsimjoin.utils.token_ordering import order_using_token_ordering


class PositionIndex(Index):
    """Builds a position index on the input column in the input table.                                                                  
                                                                                
    Position index is used by position filter, dice join, cosine join and 
    jaccard join.               
    """   

    def __init__(self, table, index_attr, tokenizer, 
                 sim_measure_type, threshold, token_ordering):
        self.table = table
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.sim_measure_type = sim_measure_type
        self.threshold = threshold
        self.token_ordering = token_ordering
        self.index = None
        self.size_cache = None
        self.min_length = maxsize
        self.max_length = 0
        super(self.__class__, self).__init__()

    def build(self, cache_empty_records=True, cache_tokens=False):
        """Build position index."""
        self.index = {}
        self.size_cache = []
        cached_tokens = []
        empty_records = []
        row_id = 0
        for row in self.table:
            # tokenize string and order the tokens using the token ordering     
            index_string = row[self.index_attr]
            index_attr_tokens = order_using_token_ordering(
                self.tokenizer.tokenize(index_string), self.token_ordering)

            # compute prefix length
            num_tokens = len(index_attr_tokens)
            prefix_length = get_prefix_length(
                                num_tokens,
                                self.sim_measure_type, self.threshold,
                                self.tokenizer)

            # update the index
            pos = 0
            for token in index_attr_tokens[0:prefix_length]:
                if self.index.get(token) is None:
                    self.index[token] = []
                self.index.get(token).append((row_id, pos))
                pos += 1

            self.size_cache.append(num_tokens)

            # keep track of the max size and min size.
            if num_tokens < self.min_length:
                self.min_length = num_tokens

            if num_tokens > self.max_length:
                self.max_length = num_tokens

            # if cache_tokens flag is set to True, the store the tokens. 
            if cache_tokens:
                cached_tokens.append(index_attr_tokens)

            if cache_empty_records and num_tokens == 0:
                empty_records.append(row_id)

            row_id += 1            

        return {'cached_tokens' : cached_tokens,
                'empty_records' : empty_records}

    def probe(self, token):
        """Probe position index using the input token."""
        return self.index.get(token, [])

    def get_size(self, row_id):
        return self.size_cache[row_id]
