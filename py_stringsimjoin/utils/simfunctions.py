from py_stringmatching.simfunctions import cosine
from py_stringmatching.simfunctions import jaccard


def get_sim_function(sim_measure_type):
    """Obtain a similarity function.

    Args:
        sim_measure_type : String, similarity measure type ('JACCARD', 'COSINE', 'DICE', 'OVERLAP')

    Returns:
        similarity function

    Examples:
        >>> jaccard_fn = get_sim_function('JACCARD')
    """
    if sim_measure_type == 'COSINE':
        return cosine
    elif sim_measure_type == 'JACCARD':
        return jaccard
