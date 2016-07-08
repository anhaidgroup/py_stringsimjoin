from sys import maxsize

from py_stringsimjoin.index.index import Index


class SizeIndex(Index):
    """Builds an index on the number of tokens in the input column in the 
    input table.                                                                  
                                                                                
    Size index is used by size filter.                                                               
    """  

    def __init__(self, table, index_attr, tokenizer):
        self.table = table
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.index = None
        self.min_length = maxsize
        self.max_length = 0
        super(self.__class__, self).__init__()

    def build(self, cache_empty_records=True):
        """Build size index."""
        self.index = {}
        empty_records = []
        row_id = 0
        for row in self.table:
            # tokenize string and compute the number of tokens
            index_string = row[self.index_attr]
            num_tokens = len(self.tokenizer.tokenize(index_string))

            # keep track of max size and min size.
            if num_tokens < self.min_length:
                self.min_length = num_tokens

            if num_tokens > self.max_length:
                self.max_length = num_tokens

            if cache_empty_records and num_tokens == 0:
                empty_records.append(row_id)

            # do not index empty records.           
            if num_tokens == 0:
                row_id += 1
                continue
 
            # update index
            if self.index.get(num_tokens) is None:
                self.index[num_tokens] = []
            self.index.get(num_tokens).append(row_id)

            row_id += 1  

        return {'empty_records': empty_records}

    def probe(self, num_tokens):
        """Probe size index using the input size."""
        return self.index.get(num_tokens, [])
