
import unittest

from nose.tools import assert_equal, assert_list_equal, raises
import pandas as pd

from py_stringsimjoin.profiler.profiler import profile_table_for_join


class ProfileTableForJoinTestCases(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame([('1', 'data science'),
                                   ('2', None),
                                   ('3', 'data integration'),
                                   ('4', ''),
                                   ('5', 'data science')],
                                   columns = ['id', 'attr'])

    def test_profile_table_for_join(self):
        profile_output = profile_table_for_join(self.table)

        expected_output_attrs = ['Unique values', 'Missing values', 'Comments']
        # verify whether the output dataframe has the necessary attributes.
        assert_list_equal(list(profile_output.columns.values),
                          expected_output_attrs)
        
        expected_unique_column = ['5 (100.0%)', '4 (80.0%)']
        # verify whether correct values are present in 'Unique values' column.
        assert_list_equal(list(profile_output['Unique values']),
                          expected_unique_column)

        expected_missing_column = ['0 (0.0%)', '1 (20.0%)']
        # verify whether correct values are present in 'Missing values' column.
        assert_list_equal(list(profile_output['Missing values']),
                          expected_missing_column)

        expected_comments = ['This attribute can be used as a key attribute.',
                             'Joining on this attribute will ignore 1 (20.0%) rows.']
        # verify whether correct values are present in 'Comments' column.
        assert_list_equal(list(profile_output['Comments']),
                          expected_comments)

        # verify whether index name is set correctly in the output dataframe.
        assert_equal(profile_output.index.name, 'Attribute')

        expected_index_column = ['id', 'attr']
        # verify whether correct values are present in the dataframe index.
        assert_list_equal(list(profile_output.index.values),
                          expected_index_column)

    def test_profile_table_for_join_with_profile_attrs(self):
        profile_output = profile_table_for_join(self.table, ['attr'])

        expected_output_attrs = ['Unique values', 'Missing values', 'Comments']
        # verify whether the output dataframe has the necessary attributes.
        assert_list_equal(list(profile_output.columns.values),
                          expected_output_attrs)

        expected_unique_column = ['4 (80.0%)']
        # verify whether correct values are present in 'Unique values' column.
        assert_list_equal(list(profile_output['Unique values']),
                          expected_unique_column)

        expected_missing_column = ['1 (20.0%)']
        # verify whether correct values are present in 'Missing values' column.
        assert_list_equal(list(profile_output['Missing values']),
                          expected_missing_column)

        expected_comments = ['Joining on this attribute will ignore 1 (20.0%) rows.']
        # verify whether correct values are present in 'Comments' column.
        assert_list_equal(list(profile_output['Comments']),
                          expected_comments)

        # verify whether index name is set correctly in the output dataframe.
        assert_equal(profile_output.index.name, 'Attribute')

        expected_index_column = ['attr']
        # verify whether correct values are present in the dataframe index.
        assert_list_equal(list(profile_output.index.values),
                          expected_index_column)

    @raises(TypeError)
    def test_profile_table_for_join_invalid_table(self):
        profile_table_for_join([])

    @raises(AssertionError)
    def test_profile_table_for_join_invalid_profile_attr(self):
        profile_table_for_join(self.table, ['id', 'invalid_attr'])

