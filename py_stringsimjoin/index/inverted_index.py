from py_stringsimjoin.index.index import Index


class InvertedIndex(Index):
    def __init__(self, table, key_attr, index_attr, tokenizer):
        self.table = table
        self.key_attr = key_attr
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.index = {}
        super(self.__class__, self).__init__()

    def build(self):
        for row in self.table:
            index_string = str(row[self.index_attr])
            # check for empty string
            if not index_string:
                continue
            index_attr_tokens = set(self.tokenizer(index_string))

            row_id = row[self.key_attr]
            for token in index_attr_tokens:
                if self.index.get(token) is None:
                    self.index[token] = []
                self.index.get(token).append(row_id)

        return True

    def probe(self, token):
        return self.index.get(token, [])
