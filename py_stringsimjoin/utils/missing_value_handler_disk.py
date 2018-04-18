import pandas as pd
import pyprind
import os

from py_stringsimjoin.utils.generic_helper import \
    find_output_attribute_indices, get_output_header_from_tables, \
    get_output_row_from_tables


missing_pairs_output = "missing_pairs.csv"
def get_pairs_with_missing_value_disk(ltable, rtable,
                                 l_key_attr, r_key_attr,
                                 l_join_attr, r_join_attr,
                                 l_out_attrs=None, r_out_attrs=None,
                                 l_out_prefix='l_', r_out_prefix='r_',
                                 out_sim_score=False, show_progress=True, global_path = os.getcwd(), data_limit = 100000):

    # find column indices of key attr, join attr and output attrs in ltable
    l_columns = list(ltable.columns.values)
    l_key_attr_index = l_columns.index(l_key_attr)
    l_join_attr_index = l_columns.index(l_join_attr)
    l_out_attrs_indices = find_output_attribute_indices(l_columns, l_out_attrs)

    # find column indices of key attr, join attr and output attrs in rtable
    r_columns = list(rtable.columns.values)
    r_key_attr_index = r_columns.index(r_key_attr)
    r_join_attr_index = r_columns.index(r_join_attr)
    r_out_attrs_indices = find_output_attribute_indices(r_columns, r_out_attrs)
   
    # find ltable records with missing value in l_join_attr
    ltable_missing = ltable[pd.isnull(ltable[l_join_attr])]

    # find ltable records which do not contain missing value in l_join_attr
    ltable_not_missing = ltable[pd.notnull(ltable[l_join_attr])]

    # find rtable records with missing value in r_join_attr
    rtable_missing = rtable[pd.isnull(rtable[r_join_attr])]

    output_rows = []
    has_output_attributes = (l_out_attrs is not None or
                             r_out_attrs is not None)

    if show_progress:
        print('Finding pairs with missing value...')
        prog_bar = pyprind.ProgBar(len(ltable_missing) + len(rtable_missing))

    missing_pairs_output_path = os.path.join(global_path,missing_pairs_output)
    if os.path.isfile(missing_pairs_output_path):
        os.remove(missing_pairs_output_path)

    # For each ltable record with missing value in l_join_attr,
    # output a pair corresponding to every record in rtable.
    for l_row in ltable_missing.itertuples(index=False):
        for r_row in rtable.itertuples(index=False):
            if has_output_attributes:
                output_row = get_output_row_from_tables(
                                 l_row, r_row,
                                 l_key_attr_index, r_key_attr_index,
                                 l_out_attrs_indices, r_out_attrs_indices)
            else:
                output_row = [l_row[l_key_attr_index], r_row[r_key_attr_index]]
            output_rows.append(output_row)

            # Flushing the data onto the disk if in-memory size exceeds the permissible data limit
            if len(output_rows) > data_limit:
                df = pd.DataFrame(output_rows)
                with open(missing_pairs_output_path, 'a+') as myfile:
                    df.to_csv(myfile, header=False, index=False)
                output_rows = []

        if show_progress:
            prog_bar.update()

    # if output rows have some data left, flush the same to the disk to maintain consistency.
    if len(output_rows) > 0:
        df = pd.DataFrame(output_rows)
        with open(missing_pairs_output_path, 'a+') as myfile:
            df.to_csv(myfile, header=False, index=False)
        output_rows = []

    # For each rtable record with missing value in r_join_attr,
    # output a pair corresponding to every record in ltable which 
    # doesn't have a missing value in l_join_attr.
    for r_row in rtable_missing.itertuples(index=False):
        for l_row in ltable_not_missing.itertuples(index=False):
            if has_output_attributes:
                output_row = get_output_row_from_tables(
                                 l_row, r_row,
                                 l_key_attr_index, r_key_attr_index,
                                 l_out_attrs_indices, r_out_attrs_indices)
            else:
                output_row = [l_row[l_key_attr_index], r_row[r_key_attr_index]]

            if out_sim_score:
                output_row.append(pd.np.NaN)

            output_rows.append(output_row)

            # Flushing the data onto the disk if in-memory size exceeds the permissible data limit
            if len(output_rows) > data_limit:
                df = pd.DataFrame(output_rows)
                with open(missing_pairs_output_path, 'a+') as myfile:
                    df.to_csv(myfile, header=False, index=False)
                output_rows = []

        if show_progress:
            prog_bar.update()

    # if output rows have some data left, flush the same to the disk to maintain consistency.
    if len(output_rows) > 0:
        df = pd.DataFrame(output_rows)
        with open(missing_pairs_output_path, 'a+') as myfile:
            df.to_csv(myfile, header=False, index=False)
        output_rows = []

    output_header = get_output_header_from_tables(
                        l_key_attr, r_key_attr,
                        l_out_attrs, r_out_attrs,
                        l_out_prefix, r_out_prefix)

    if out_sim_score:
        output_header.append("_sim_score")

    return missing_pairs_output_path
