import pyprind


class Filter(object):
    """Filter base class.
    """
    def filter_candset(self, candset,
                       candset_l_key_attr, candset_r_key_attr,
                       ltable, rtable,
                       l_key_attr, r_key_attr,
                       l_filter_attr, r_filter_attr):
        """Filter candidate set.

        Args:
        candset : Pandas data frame
        candset_l_key_attr, candset_r_key_attr : String, key attributes in candset (that refer to ltable and rtable)
        ltable, rtable : Pandas data frame, base tables from which candset was obtained
        l_key_attr, r_key_attr : String, key attribute from ltable and rtable
        l_filter_attr, r_filter_attr : String, filter attribute from ltable and rtable

        Returns:
        result : Pandas data frame
        """
        # check for empty candset
        if candset.empty:
            return candset

        # find column indices of id attr and filter attr in ltable
        l_columns = list(ltable.columns.values)
        l_key_attr_index = l_columns.index(l_key_attr)
        l_filter_attr_index = l_columns.index(l_filter_attr)

        # find column indices of id attr and filter attr in rtable
        r_columns = list(rtable.columns.values)
        r_key_attr_index = r_columns.index(r_key_attr)
        r_filter_attr_index = r_columns.index(r_filter_attr)

        # build a dictionary on ltable
        ltable_dict = {}
        for l_row in ltable.itertuples(index=False):
            ltable_dict[l_row[l_key_attr_index]] = l_row

        # build a dictionary on rtable
        rtable_dict = {}
        for r_row in rtable.itertuples(index=False):
            rtable_dict[r_row[r_key_attr_index]] = r_row

        # find indices of l_key_attr and r_key_attr in candset
        candset_columns = list(candset.columns.values)
        candset_l_key_attr_index = candset_columns.index(candset_l_key_attr)
        candset_r_key_attr_index = candset_columns.index(candset_r_key_attr)

        valid_rows = []
        prog_bar = pyprind.ProgBar(len(candset))

        for candset_row in candset.itertuples(index = False):
            l_id = candset_row[candset_l_key_attr_index]
            r_id = candset_row[candset_r_key_attr_index]

            l_row = ltable_dict[l_id]
            r_row = rtable_dict[r_id]
            valid_rows.append(not self.filter_pair(
                                      l_row[l_filter_attr_index],
                                      r_row[r_filter_attr_index]))

            prog_bar.update()

        return candset[valid_rows]
