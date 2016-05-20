from .data_generator import generate_table
from .data_generator import generate_tokens  
from py_stringsimjoin.join.join import jaccard_join
from py_stringsimjoin.utils.tokenizers import create_delimiter_tokenizer

TOKEN_LENGTH_DISTRIBUTION = {3: 0.2,
                             4: 0.2,
                             5: 0.2,
                             6: 0.2,
                             7: 0.2}

STRING_SIZE_DISTRIBUTION = {3: 0.2,
                            4: 0.2,
                            5: 0.2,
                            6: 0.2,
                            7: 0.2}

NUM_TOKENS = 500
TOKENS = generate_tokens(TOKEN_LENGTH_DISTRIBUTION, NUM_TOKENS)
NUM_RECORDS = 10000
LTABLE = generate_table(STRING_SIZE_DISTRIBUTION, TOKENS, NUM_RECORDS,
                       'id', 'attr')
RTABLE = LTABLE

class JaccardJoinBenchmark:
    def time_small_small_delim(self):
        dl = create_delimiter_tokenizer()
        jaccard_join(LTABLE, RTABLE, 'id', 'id', 'attr', 'attr', dl, 0.8)
