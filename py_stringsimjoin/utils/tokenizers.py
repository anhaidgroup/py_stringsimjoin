from functools import partial

from py_stringmatching.tokenizers import delimiter
from py_stringmatching.tokenizers import qgram


def create_delimiter_tokenizer(delim_str=' '):
    """Creates a delimiter based tokenizer using the given delimiter.

    Args:
        delim_str (str): Delimiter string

    Returns:
        tokenizer function

    Examples:
        >>> delim_tokenizer = create_delimiter_tokenizer(',')
    """
    return partial(delimiter, delim_str=delim_str)


def create_qgram_tokenizer(qval=2):
    """Creates a qgram based tokenizer using the given q value.

    Args:
        qval (int): Q-gram length (defaults to 2)

    Returns:
        tokenizer function

    Examples:
        >>> qg_tokenizer = create_qgram_tokenizer(3)
    """
    return partial(qgram, qval=qval)
