def get_output_row_from_tables(candset_id,
                               l_row, r_row,
                               l_id, r_id, 
                               l_out_attrs=None, r_out_attrs=None):
    output_row = []
    
    # add candset id
    output_row.append(candset_id)

    # add ltable id attr
    output_row.append(l_id)

    # add ltable output attributes
    if l_out_attrs:
        for l_attr in l_out_attrs:
            output_row.append(l_row[l_attr])

    # add rtable id attr
    output_row.append(r_id)

    # add rtable output attributes
    if r_out_attrs:
        for r_attr in r_out_attrs:
            output_row.append(r_row[r_attr])

    return output_row


def get_output_row_from_candset(row_dict, out_attrs):
    output_row = []

    for attr in out_attrs:
        output_row.append(row_dict[attr])

    return output_row


def get_output_header_from_tables(candset_key_attr,
                                  l_key_attr, r_key_attr,
                                  l_out_attrs, r_out_attrs,
                                  l_out_prefix, r_out_prefix):
    output_header = []

    output_header.append(candset_key_attr)

    output_header.append(l_out_prefix + l_key_attr)

    if l_out_attrs:
        for l_attr in l_out_attrs:
            output_header.append(l_out_prefix + l_attr)

    output_header.append(r_out_prefix + r_key_attr)

    if r_out_attrs:
        for r_attr in r_out_attrs:
            output_header.append(r_out_prefix + r_attr)

    return output_header


def build_dict_from_table(table, key_attr_index):
    table_dict = {}
    for row in table.itertuples(index=False):
        table_dict[row[key_attr_index]] = row
    return table_dict


def find_output_attribute_indices(original_columns, output_attributes):
    output_attribute_indices = []
    if output_attributes is not None:
        for attr in output_attributes:
            output_attribute_indices.append(original_columns.index(attr))
    return output_attribute_indices