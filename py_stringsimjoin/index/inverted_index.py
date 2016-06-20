from py_stringsimjoin.index.index import Index


class InvertedIndex(Index):
    def __init__(self, table, key_attr, index_attr, tokenizer, 
                 cache_size_map=False):
        self.table = table
        self.key_attr = key_attr
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.index = {}
        self.size_map = {}
        self.cache_size_map = cache_size_map
        super(self.__class__, self).__init__()

    def build(self):
        for row in self.table:
            index_string = str(row[self.index_attr])
            index_attr_tokens = self.tokenizer.tokenize(index_string)

            row_id = row[self.key_attr]
            for token in index_attr_tokens:
                if self.index.get(token) is None:
                    self.index[token] = []
                self.index.get(token).append(row_id)

            if self.cache_size_map:
                self.size_map[row_id] = len(index_attr_tokens)

        return True

    def probe(self, token):
        return self.index.get(token, [])
