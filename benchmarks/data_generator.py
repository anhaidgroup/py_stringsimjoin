import random
import string

import pandas as pd


def generate_tokens(length_distribution, num_tokens):
    lengths = []
    cum_prob = []
    i = 0
    for length in length_distribution:
        lengths.append(length)
        if i == 0:
            cum_prob.append(length_distribution[length])
        else:
            cum_prob.append(length_distribution[length] + cum_prob[i-1])
        i += 1
    tokens = {}
    cnt = 0
    while cnt < num_tokens:
        rand = random.random()
        selected_length = -1
        for i in range(len(cum_prob)):
            if rand <= cum_prob[i]:
                selected_length = lengths[i]
                break
        flag = True
        while flag:
            new_token = ''.join(random.choice(string.ascii_lowercase)
                                for i in range(selected_length))
            if tokens.get(new_token) == None:
                tokens[new_token] = True
                flag = False
        cnt += 1
    return list(tokens.keys())


def generate_table(size_distribution, tokens, num_records,
                   id_col_name, attr_col_name):
    sizes = []
    cum_prob = []
    i = 0
    for size in size_distribution:
        sizes.append(size)
        if i == 0:
            cum_prob.append(size_distribution[size])
        else:
            cum_prob.append(size_distribution[size] + cum_prob[i-1])
        i += 1

    records = []
    cnt = 0
    num_tokens = len(tokens)
    while cnt < num_records:
        rand = random.random()
        selected_size = -1
        for i in range(len(cum_prob)):
            if rand <= cum_prob[i]:
                selected_size = sizes[i]
                break
        new_string = ''
        for i in range(selected_size):
            rand = random.randint(0, num_tokens - 1)
            if i == 0:
                new_string += tokens[rand]
            else:
                new_string += ' ' + tokens[rand]

        records.append([cnt, new_string])
        cnt += 1
    return pd.DataFrame(records, columns=[id_col_name, attr_col_name])
