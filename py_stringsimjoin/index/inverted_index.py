from py_stringsimjoin.index.index import Index


class InvertedIndex(Index):
    def __init__(self, table, index_attr, tokenizer, 
                 cache_size_flag=False):
        self.table = table
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.index = {}
        self.size_cache = []
        self.cache_size_flag = cache_size_flag
        super(self.__class__, self).__init__()

    def build(self, cache_empty_records=True):
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
        return self.index.get(token, [])
