from py_stringsimjoin.index.index import Index

class SizeIndex(Index):

    def __init__(self, table, id_attr, index_attr, tokenizer):
        self.table = table
        self.id_attr = id_attr
        self.index_attr = index_attr
        self.tokenizer = tokenizer
        self.index = {}
        self.min_length = 0
        self.max_length = 0
        super(self.__class__, self).__init__()


    def build(self):
        for row in self.table:
            num_tokens = len(set(self.tokenizer(str(row[self.index_attr]))))
            
            if self.index.get(num_tokens) is None:
                self.index[num_tokens] = []

            self.index.get(num_tokens).append(row[self.id_attr])

            if num_tokens < self.min_length:
                self.min_length = num_tokens
 
            if num_tokens > self.max_length:
                self.max_length = num_tokens  

        return True

    def probe(self, num_tokens):
        return self.index.get(num_tokens, [])
