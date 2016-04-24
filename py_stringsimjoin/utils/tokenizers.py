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
    def delimiter_tokenizer(string):
        return delimiter(string, delim_str)

    return delimiter_tokenizer


def create_qgram_tokenizer(qval=2):
    """Creates a qgram based tokenizer using the given q value.

    Args:
        qval (int): Q-gram length (defaults to 2)

    Returns:
        tokenizer function

    Examples:
        >>> qg_tokenizer = create_qgram_tokenizer(3)
    """
    def qgram_tokenizer(string):
        return qgram(string, qval)

    return qgram_tokenizer
