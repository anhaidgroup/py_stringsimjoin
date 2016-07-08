from py_stringsimjoin.index.index import Index


class InvertedIndex(Index):
    """Builds an inverted index on the input column in the input table.                                                                  
                                                             
    Inverted index is used by overlap filter and overlap coefficient join.
                                                                                
    Args:                                                                       
        table (list): Input table as list of tuples.
        index_attr (int): Position of the column in the tuple, on which inverted
            index has to be built.
        tokenizer (Tokenizer object): Tokenizer used to tokenize the index_attr.
        cache_size_flag (boolean): A flag indicating whether size of column
            needs to be cached in the index.  
                                                                                
    Attributes:                                                                 
        table (list): An attribute to store the input table.                            
        index_attr (int): An attribute to store the position of the column in
            the tuple.                                              
        tokenizer (Tokenizer object): An attribute to store the tokenizer.
        cache_size_flag (boolean): An attribute to store the value of the flag
            cache_size_flag.
        index (dict): A Python dictionary storing the inverted index where the
            key is the token and value is a list of tuple ids. Currently, we use
            the index of the tuple in the table as its id.
    """ 

    def __init__(self, table, index_attr, tokenizer, 
                 cache_size_flag=False):
        
        self.table = table
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.cache_size_flag = cache_size_flag 
        self.index = None
        self.size_cache = None
        super(self.__class__, self).__init__()

    def build(self, cache_empty_records=True):
        """Build inverted index."""
        self.index = {}
        self.size_cache = []
        empty_records = []
        row_id = 0
        for row in self.table:
            index_string = row[self.index_attr]
            index_attr_tokens = self.tokenizer.tokenize(index_string)

            for token in index_attr_tokens:
                if self.index.get(token) is None:
                    self.index[token] = []
                self.index.get(token).append(row_id)

            num_tokens = len(index_attr_tokens)
            if self.cache_size_flag:
                self.size_cache.append(num_tokens)

            if cache_empty_records and num_tokens == 0:
                empty_records.append(row_id)

            row_id += 1

        return {'empty_records': empty_records}

    def probe(self, token):
        """Probe the index using the input token."""
        return self.index.get(token, [])
