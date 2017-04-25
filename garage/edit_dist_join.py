
from os.path import dirname, join

from py_stringmatching import QgramTokenizer

import py_stringsimjoin as ssj
import py_stringmatching as sm

def flush_dataset(output_pairs):
    base_dir = join(dirname(__file__), 'data')
    output_pairs.to_csv(join(base_dir, 'output.csv'),index=None)

A, B = ssj.load_person_dataset()


ws = sm.WhitespaceTokenizer(return_set=True)


#output_pairs = \
ssj.edit_distance_join(A, B, 'A.ID', 'B.ID', 'A.name', 'B.name', 0.3,

 '<=',   False,
                             ['A.birth_year', 'A.zipcode'],
                             ['B.name', 'B.zipcode'],
                             'ltable.', 'rtable.',
                             True, 3, True,
                                tokenizer=QgramTokenizer(qval=2)
                          #     , None
                                , file_name='data/output.csv'
                                , mem_threshold=-1

                                )

#for n_jobs in [-1, 0, 1, 3]:
#    for mem_thres in [-1, 0, 1e9, 2e9]:
#        for file_name in [None, 'output.csv']:
#            from py_stringsimjoin.join.edit_distance_join import edit_distance_join

#            test_function = partial(test_valid_join3, test_scenario_1,
#                                    edit_distance_join,
#                                    (QgramTokenizer(qval=2),
#                                     0.3, '<=', True, False,
#                                     ['A.birth_year', 'A.zipcode'],
#                                     ['B.name', 'B.zipcode'],
#                                     'ltable.', 'rtable.',
#                                     True, n_jobs, True, file_name
#                                     , mem_thres))
#            test_function.description = 'Test Tuple Pair Chest '
#            yield test_function,
#flush_dataset(output_pairs)
print('Done !!!')