"""Custom benchmarking module"""
import json
import os
import time

from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
import pandas as pd

from py_stringsimjoin.join.cosine_join import cosine_join
from py_stringsimjoin.join.dice_join import dice_join
from py_stringsimjoin.join.edit_distance_join import edit_distance_join
from py_stringsimjoin.join.jaccard_join import jaccard_join
from py_stringsimjoin.join.overlap_coefficient_join import overlap_coefficient_join
from py_stringsimjoin.join.overlap_join import overlap_join
from py_stringsimjoin.utils.helper_functions import get_install_path


JOIN_FUNCTIONS = {'COSINE': cosine_join,
                  'DICE': dice_join,
                  'EDIT_DISTANCE': edit_distance_join,
                  'JACCARD': jaccard_join,
                  'OVERLAP': overlap_join,
                  'OVERLAP_COEFFICIENT': overlap_coefficient_join}

TOKENIZERS = {'SPACE_DELIMITER': DelimiterTokenizer(delim_set=[' '],
                                                    return_set=True),
              '2_GRAM': QgramTokenizer(qval=2, return_set=True),
              '3_GRAM': QgramTokenizer(qval=3, return_set=True),
              '2_GRAM_BAG': QgramTokenizer(qval=2),
              '3_GRAM_BAG': QgramTokenizer(qval=3)}

# path where datasets are present
BASE_PATH = os.sep.join([get_install_path(), 'benchmarks', 'example_datasets'])

# join scenarios json file. If you need to perform benchmark on a new dataset,
# add a entry for that dataset in the json file.
JOIN_SCENARIOS_FILE = 'join_scenarios.json'

# scenarios that need to be skipped from benchmarking
EXCLUDE_SCENARIOS = ["restaurants_edit_dist", "music_edit_dist"]

# number of times to run each benchmark
NUMBER_OF_EXECUTIONS = 1

# benchmark output directory
OUTPUT_DIR = '_benchmark_results'

class JoinScenario:
    def __init__(self, scenario_name, ltable, rtable, l_id_attr, r_id_attr,
                 l_join_attr, r_join_attr, tokenizers,
                 sim_measure_types, thresholds, n_jobs):
        self.scenario_name = scenario_name 
        self.ltable = os.sep.join(ltable)
        self.rtable = os.sep.join(rtable)
        self.l_id_attr = l_id_attr
        self.r_id_attr = r_id_attr
        self.l_join_attr = l_join_attr
        self.r_join_attr = r_join_attr
        self.tokenizers = tokenizers
        self.sim_measure_types = sim_measure_types
        self.thresholds = thresholds
        self.n_jobs = n_jobs


def load_join_scenarios():
    fp = open(JOIN_SCENARIOS_FILE, 'r')
    scenarios_json = json.load(fp)['scenarios']
    fp.close()
    join_scenarios = []
    for sc in scenarios_json.keys():
        join_scenario = JoinScenario(sc, scenarios_json[sc]['ltable'],
                                     scenarios_json[sc]['rtable'],
                                     scenarios_json[sc]['l_id_attr'],
                                     scenarios_json[sc]['r_id_attr'],
                                     scenarios_json[sc]['l_join_attr'],
                                     scenarios_json[sc]['r_join_attr'],
                                     scenarios_json[sc]['tokenizers'],
                                     scenarios_json[sc]['sim_measure_types'],
                                     scenarios_json[sc]['thresholds'],
                                     scenarios_json[sc]['n_jobs'])
        join_scenarios.append(join_scenario)
    return join_scenarios


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # load scenarios
    scenarios = load_join_scenarios()

    output_header = ','.join(['similarity measure type', 'tokenizer',
                        'threshold', 'n_jobs', 'candset size', 'avg time']) 

    for scenario in scenarios:
        if scenario.scenario_name in EXCLUDE_SCENARIOS:
            continue
        ltable_path = os.sep.join([BASE_PATH, scenario.ltable])
        rtable_path = os.sep.join([BASE_PATH, scenario.rtable])
        if not os.path.exists(ltable_path):
            print('Left table not found for scenario \'', scenario.scenario_name, '\': ', ltable_path)
            print('Skipping scenario for benchmark...')

        if not os.path.exists(rtable_path): 
            print('Right table not found for scenario \'', scenario.scenario_name, '\': ', rtable_path)
            print('Skipping scenario for benchmark...')

        output_file = open(os.sep.join([OUTPUT_DIR, scenario.scenario_name]), 'w')        
        output_file.write('%s\n' % output_header)

        # load input tables for the scenario
        ltable = pd.read_csv(ltable_path)
        rtable = pd.read_csv(rtable_path)

        for sim_measure_type in scenario.sim_measure_types:
            join_fn = JOIN_FUNCTIONS[sim_measure_type]

            for tokenizer in scenario.tokenizers:
                if (sim_measure_type == 'EDIT_DISTANCE' and
                    tokenizer == 'SPACE_DELIMITER'):
                    continue
                tok = TOKENIZERS[tokenizer]

                for threshold in scenario.thresholds:
                    for n_jobs in scenario.n_jobs:
                        if sim_measure_type ==  'EDIT_DISTANCE':
                            args = (threshold, None, None, 'l_', 'r_', True,
                                    n_jobs, tok) 
                        else:
                            args = (tok, threshold, None, None, 'l_', 'r_',
                                    True, n_jobs)

                        cumulative_time = 0
                        candset_size = 0
                        for i in range(NUMBER_OF_EXECUTIONS):
                            start_time = time.time()
                            C = join_fn(ltable, rtable, scenario.l_id_attr, scenario.r_id_attr, scenario.l_join_attr, scenario.r_join_attr, *args)
                            cumulative_time += (time.time() - start_time)
                            candset_size = len(C)

                        avg_time_elapsed = float(cumulative_time) / float(NUMBER_OF_EXECUTIONS)      
                        output_record = ','.join([str(sim_measure_type), str(tokenizer),
                                                  str(threshold), str(n_jobs),
                                                  str(candset_size), str(avg_time_elapsed)])
                        output_file.write('%s\n' % output_record)
        output_file.close()
