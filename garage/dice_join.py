
from os.path import dirname, join

import py_stringsimjoin as ssj
import py_stringmatching as sm

def flush_dataset(output_pairs):
    base_dir = join(dirname(__file__), 'data')
    output_pairs.to_csv(join(base_dir, 'output.csv'),index=None)

A, B = ssj.load_person_dataset()


ws = sm.WhitespaceTokenizer(return_set=True)

#output_pairs = \
ssj.dice_join(A, B, 'A.ID', 'B.ID', 'A.name', 'B.name', ws, 0.3,

 '>=', True, False,
                             ['A.birth_year', 'A.zipcode'],
                             ['B.name', 'B.zipcode'],
                             'ltable.', 'rtable.',
                             True, 3, True
         #                       , None
                                , 'data/output.csv'
                               , -1
                                  )





#flush_dataset(output_pairs)
print('Done !!!')