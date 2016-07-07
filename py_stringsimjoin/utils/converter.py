import pandas as pd

def convert2str(dataframe, col_name, inplace=True, return_col=False):

    if not isinstance(inplace, bool):
        raise AssertionError('Parameter \'inplace\' is not of type bool')

    if not isinstance(return_col, bool):                                           
        raise AssertionError('Parameter \'return_col\' is not of type bool')

    if inplace and return_col:
        raise AssertionError('Both \'inplace\' and \'return_col\' parameters' +\
                             'cannot be set to True')

    col_type = dataframe[col_name].dtype
    if col_type == object:
        if inplace:
            return True
        elif return_col:
            return dataframe[col_name].copy()
        else:
            return dataframe.copy()
   
    if col_type == int:
        if inplace:
            dataframe[col_name] = dataframe[col_name].astype(str)
            return True
        elif return_col:
            return dataframe[col_name].astype(str)
        else:
            dataframe_copy = dataframe.copy()
            dataframe_copy[col_name] = dataframe_copy[col_name].astype(str)               
            return dataframe_copy
   
    if col_type == float:
        col_non_nan_values = dataframe[col_name].dropna()
        int_values = sum(col_non_nan_values.apply(
                                            lambda val: val - int(val) == 0))
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
