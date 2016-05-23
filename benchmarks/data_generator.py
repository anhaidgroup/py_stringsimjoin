"""Utilities to generate synthetic data"""
import random
import string

import pandas as pd


def generate_tokens(mean, std_dev, num_tokens):
    tokens = {}
    cnt = 0
    while cnt < num_tokens:
        length = int(round(random.normalvariate(mean,
                                                std_dev)))
        if length < 2:
            continue
        flag = True
        while flag:
            new_token = ''.join(random.choice(string.ascii_lowercase)
                                for i in range(length))
            if tokens.get(new_token) is None:
                tokens[new_token] = True
                flag = False
        cnt += 1
    return list(tokens.keys())


def generate_table(mean, std_dev, tokens, num_records,
                   id_col_name, attr_col_name):
    records = []
    cnt = 0
    num_tokens = len(tokens)
    while cnt < num_records:
        size = int(round(random.normalvariate(mean,
                                              std_dev)))
        new_string = ''
        for i in range(size):
            rand = random.randint(0, num_tokens - 1)
            if i == 0:
                new_string += tokens[rand]
            else:
                new_string += ' ' + tokens[rand]

        records.append([cnt, new_string])
        cnt += 1
    return pd.DataFrame(records, columns=[id_col_name, attr_col_name])
