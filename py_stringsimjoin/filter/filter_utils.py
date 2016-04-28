from math import ceil
from math import floor
from math import sqrt
from sys import maxint


def get_size_lower_bound(num_tokens, sim_measure_type, threshold):
    if sim_measure_type == 'COSINE':
        return int(ceil(threshold * threshold * num_tokens))
    elif sim_measure_type == 'DICE':
        return int(ceil((threshold / (2 - threshold)) * num_tokens))
    elif sim_measure_type == 'JACCARD':
        return int(ceil(threshold * num_tokens))
    elif sim_measure_type == 'OVERLAP':
        return threshold


def get_size_upper_bound(num_tokens, sim_measure_type, threshold):
    if sim_measure_type == 'COSINE':
        return int(floor(num_tokens / (threshold * threshold)))
    elif sim_measure_type == 'DICE':
        return int(floor(((2 - threshold) / threshold) * num_tokens))
    elif sim_measure_type == 'JACCARD':
        return int(floor(num_tokens / threshold))
    elif sim_measure_type == 'OVERLAP':
        return maxint


def get_prefix_length(num_tokens, sim_measure_type, threshold):
    if sim_measure_type == 'COSINE':
        return int(num_tokens -
                   ceil(threshold * threshold * num_tokens) + 1)
    elif sim_measure_type == 'DICE':
        return int(num_tokens -
                   ceil((threshold / (2 - threshold)) * num_tokens) + 1)
    elif sim_measure_type == 'JACCARD':
        return int(num_tokens - ceil(threshold * num_tokens) + 1)
    elif sim_measure_type == 'OVERLAP':
        return num_tokens - threshold + 1


def get_overlap_threshold(l_num_tokens, r_num_tokens,
                          sim_measure_type, threshold):
    if sim_measure_type == 'COSINE':
        return ceil(threshold * sqrt(l_num_tokens * r_num_tokens))
    elif sim_measure_type == 'DICE':
        return ceil((threshold / 2) * (l_num_tokens + r_num_tokens))
    elif sim_measure_type == 'JACCARD':
        return ceil((threshold / (1 + threshold)) * (l_num_tokens + r_num_tokens))
    elif sim_measure_type == 'OVERLAP':
        return threshold
