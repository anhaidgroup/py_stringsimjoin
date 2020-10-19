import numpy as np
import pandas as pd

def dataframe_column_to_str(dataframe, col_name, inplace=False, 
                            return_col=False):
    """Convert columun in the dataframe into string type while preserving NaN 
    values.

    This method is useful when performing join over numeric columns. Currently, 
    the join methods expect the join columns to be of string type. Hence, the 
    numeric columns need to be converted to string type before performing the 
    join. 
 
    Args:
        dataframe (DataFrame): Input pandas dataframe.
        col_name (string): Name of the column in the dataframe to be converted.
        inplace (boolean): A flag indicating whether the input dataframe should 
            be modified inplace or in a copy of it.
        return_col (boolean): A flag indicating whether a copy of the converted
            column should be returned. When this flag is set to True, the method
            will not modify the original dataframe and will return a new column
            of string type. Only one of inplace and return_col can be set to 
            True.
    
    Returns:
        A Boolean value when inplace is set to True.

        A new dataframe when inplace is set to False and return_col is set to False.

        A series when inplace is set to False and return_col is set to True. 
    """

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

    if inplace:
        num_rows = len(dataframe[col_name])
        if (num_rows == 0 or sum(pd.isnull(dataframe[col_name])) == num_rows):
            dataframe[col_name] = dataframe[col_name].astype(np.object)
            return True
        else:
            return series_to_str(dataframe[col_name], inplace)
    elif return_col:
        return series_to_str(dataframe[col_name], inplace)
    else:
        dataframe_copy = dataframe.copy()
        series_to_str(dataframe_copy[col_name], True)
        return dataframe_copy
        

def series_to_str(series, inplace=False):
    """Convert series into string type while preserving NaN values.                                                                     
                                                                                
    Args:                                                                       
        series (Series): Input pandas series.                                 
        inplace (boolean): A flag indicating whether the input series should 
            be modified inplace or in a copy of it. This flag is ignored when
            the input series consists of only NaN values or the series is 
            empty (with int or float type). In these two cases, we always return
            a copy irrespective of the inplace flag.                        
                                                                                
    Returns:                                                                    
        A Boolean value when inplace is set to True.                            

        A series when inplace is set to False.    
    """
   
    if not isinstance(series, pd.Series):                                 
        raise AssertionError('First argument is not of type pandas dataframe')

    if not isinstance(inplace, bool):                                           
        raise AssertionError('Parameter \'inplace\' is not of type bool')
    
    col_type = series.dtype                                                     

    # Currently, we ignore the inplace flag when the series is empty and is of
    # type int or float. In this case, we will always return a copy.                       
    if len(series) == 0:                                                        
        if col_type == np.object and inplace:
            return True
        else:                                                                      
            return series.astype(np.object)    

    if col_type == np.object:
        # If column is already of type object, do not perform any conversion.
        if inplace:
            return True
        else:
            return series.copy()
    elif np.issubdtype(col_type, np.integer):
        # If the column is of type int, then there are no missing values in the 
        # column and hence we can directly convert it to string using           
        # the astype method.     
        col_str = series.astype(str)
        if inplace:
            series.update(col_str)
            return True
        else:
            return col_str
    elif np.issubdtype(col_type, np.float):
        # If the column is of type float, then there are two cases:             
        # (1) column only contains interger values along with NaN.              
        # (2) column actually contains floating point values.                   
        # For case 1, we preserve the NaN values as such and convert the float  
        # values to string by first converting them to int and then to string.  
        # For case 1, we preserve the NaN values as such and convert the float  
        # values directly to string.   

        # get the column values that are not NaN                                    
        col_non_nan_values = series.dropna()
 
        # Currently, we ignore the inplace flag when all values in the column
        # are NaN and will always return a copy of the column cast into 
        # object type.
        if len(col_non_nan_values) == 0:
            return series.astype(np.object)
     
        # find how many of these values are actually integer values cast into   
        # float.
        int_values = sum(col_non_nan_values.apply(lambda val: val.is_integer()))

        # if all these values are interger values, then we handle according     
        # to case 1, else we proceed by case 2. 
        if int_values == len(col_non_nan_values):                               
            col_str = series.apply(lambda val: np.NaN if        
                                            pd.isnull(val) else str(int(val)))  
        else:                                                                   
            col_str = series.apply(lambda val: np.NaN if        
                                            pd.isnull(val) else str(val))
        if inplace:
            series.update(col_str)
            return True
        else:
            return col_str
    else:
        raise TypeError('Invalid column type. ' + \
                        'Cannot convert the column to string.')
