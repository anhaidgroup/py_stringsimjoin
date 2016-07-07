import pandas as pd

def convert2str(dataframe, col_name, inplace=False, return_col=False):

    if not isinstance(dataframe, pd.DataFrame):
        raise AssertionError('First argument is not of type pandas dataframe')

    if col_name not in dataframe.columns:
        raise AssertionError('Column \'' + col_name + '\' not found in the' + \
                             ' input dataframe')

    if not isinstance(inplace, bool):
        raise AssertionError('Parameter \'inplace\' is not of type bool')

    if not isinstance(return_col, bool):                                           
        raise AssertionError('Parameter \'return_col\' is not of type bool')

    if inplace and return_col:
        raise AssertionError('Both \'inplace\' and \'return_col\' parameters' +\
                             'cannot be set to True')

    col_type = dataframe[col_name].dtype

    if col_type == object:
        # If column is already of type object, do not perform any conversion.
        if inplace:
            return True
        elif return_col:
            return dataframe[col_name].copy()
        else:
            return dataframe.copy()
    elif col_type == int:
        # If the column is of type int, then there are no missing values in the
        # column and hence we can directly convert it to string using 
        # the astype method.
        if inplace:
            dataframe[col_name] = dataframe[col_name].astype(str)
            return True
        elif return_col:
            return dataframe[col_name].astype(str)
        else:
            dataframe_copy = dataframe.copy()
            dataframe_copy[col_name] = dataframe_copy[col_name].astype(str)               
            return dataframe_copy
    elif col_type == float:
        # If the column is of type float, then there are two cases:
        # (1) column only contains interger values along with NaN.
        # (2) column actually contains floating point values.
        # For case 1, we preserve the NaN values as such and convert the float
        # values to string by first converting them to int and then to string.
        # For case 1, we preserve the NaN values as such and convert the float
        # values directly to string.

        # get the column values that are not NaN
        col_non_nan_values = dataframe[col_name].dropna()
        # find how many of these values are actually integer values cast into 
        # float.
        int_values = sum(col_non_nan_values.apply(
                                            lambda val: val - int(val) == 0))
        # if all these values are interger values, then we handle according
        # to case 1, else we proceed by case 2.
        if int_values == len(col_non_nan_values):
            col_str = dataframe[col_name].apply(lambda val: pd.np.NaN if 
                                            pd.isnull(val) else str(int(val)))
        else:
            col_str = dataframe[col_name].apply(lambda val: pd.np.NaN if        
                                            pd.isnull(val) else str(val))

        if inplace:
            dataframe[col_name] = col_str
            return True
        elif return_col:
            return col_str
        else:
            dataframe_copy = dataframe.copy()                                   
            dataframe_copy[col_name] = col_str     
            return dataframe_copy
    else:
        raise TypeError('Invalid column type. ' + \
                        'Cannot convert the column to string.')           
