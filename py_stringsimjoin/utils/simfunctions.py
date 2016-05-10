from py_stringsimjoin.externals.py_stringmatching.simfunctions import cosine
from py_stringsimjoin.externals.py_stringmatching.simfunctions import jaccard
from py_stringsimjoin.externals.py_stringmatching.simfunctions import \
                                                                   levenshtein


def get_sim_function(sim_measure_type):
    """Obtain a similarity function.

    Args:
        sim_measure_type : String, similarity measure type ('JACCARD', 'COSINE', 'DICE', 'EDIT_DISTANCE', ''OVERLAP')

    Returns:
        similarity function

    Examples:
        >>> jaccard_fn = get_sim_function('JACCARD')
    """
    if sim_measure_type == 'COSINE':
        return cosine
    elif sim_measure_type == 'EDIT_DISTANCE':
        return levenshtein
    elif sim_measure_type == 'JACCARD':
        return jaccard
