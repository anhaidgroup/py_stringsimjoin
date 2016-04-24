def gen_token_ordering_for_lists(token_lists):
    token_freq_dict = {}
    for token_list in token_lists:
        for token in token_list:
            token_freq_dict[token] = token_freq_dict.get(token, 0) + 1
            order_idx = 1

    token_ordering = {}
    for token in sorted(token_freq_dict, key=token_freq_dict.get):
        token_ordering[token] = order_idx
        order_idx += 1

    return token_ordering


def gen_token_ordering_for_tables(table_list, attr_list, tokenizer):
    token_freq_dict = {}
    table_index = 0
    for table in table_list:
        for row in table:
            for token in set(tokenizer(str(row[attr_list[table_index]]))):
                token_freq_dict[token] = token_freq_dict.get(token, 0) + 1
        table_index += 1

    token_ordering = {}
    order_idx = 1
    for token in sorted(token_freq_dict, key=token_freq_dict.get):
        token_ordering[token] = order_idx
        order_idx += 1

    return token_ordering


def order_using_token_ordering(tokens, token_ordering):
    ordered_tokens = []

    for token in tokens:
        order = token_ordering.get(token)
        if order is not None:
            ordered_tokens.append(order)

    ordered_tokens.sort()

    return ordered_tokens
