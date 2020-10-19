import random
import string
import unittest

from nose.tools import assert_equal, assert_list_equal, raises
import numpy as np
import pandas as pd

from py_stringsimjoin.utils.converter import dataframe_column_to_str, \
                                             series_to_str


class DataframeColumnToStrTestCases(unittest.TestCase):
    def setUp(self):
        float_col = pd.Series(np.random.randn(10)).append(
            pd.Series([np.NaN for _ in range(10)], index=range(10, 20)))
        float_col_with_int_val = pd.Series(
                                     np.random.randint(1, 100, 10)).append(          
            pd.Series([np.NaN for _ in range(10)], index=range(10, 20)))        
        str_col = pd.Series([random.choice(string.ascii_lowercase) 
                      for _ in range(10)]).append(
            pd.Series([np.NaN for _ in range(10)], index=range(10, 20)))
        int_col = pd.Series(np.random.randint(1, 100, 20))                           
        nan_col = pd.Series([np.NaN for _ in range(20)])                

        self.dataframe = pd.DataFrame({'float_col': float_col,
                               'float_col_with_int_val': float_col_with_int_val,
                               'int_col': int_col,
                               'str_col': str_col,
                               'nan_col': nan_col})

    def test_str_col(self):
        assert_equal(self.dataframe['str_col'].dtype, object)
        out_df = dataframe_column_to_str(self.dataframe, 'str_col', 
                                         inplace=False, return_col=False)
        assert_equal(type(out_df), pd.DataFrame)
        assert_equal(out_df['str_col'].dtype, object)
        assert_equal(self.dataframe['str_col'].dtype, object)                   
        assert_equal(sum(pd.isnull(self.dataframe['str_col'])), 
                     sum(pd.isnull(out_df['str_col'])))                     

    def test_int_col(self):                                                     
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        out_df = dataframe_column_to_str(self.dataframe, 'int_col',                         
                                         inplace=False, return_col=False)                   
        assert_equal(type(out_df), pd.DataFrame)                                
        assert_equal(out_df['int_col'].dtype, object)
        assert_equal(self.dataframe['int_col'].dtype, int)                     
        assert_equal(sum(pd.isnull(out_df['int_col'])), 0)

    def test_float_col(self):                                                     
        assert_equal(self.dataframe['float_col'].dtype, float)                   
        out_df = dataframe_column_to_str(self.dataframe, 'float_col',                         
                                         inplace=False, return_col=False)                   
        assert_equal(type(out_df), pd.DataFrame)                                
        assert_equal(out_df['float_col'].dtype, object)
        assert_equal(self.dataframe['float_col'].dtype, float)                                             
        assert_equal(sum(pd.isnull(self.dataframe['float_col'])),                           
                     sum(pd.isnull(out_df['float_col']))) 

    def test_float_col_with_int_val(self):                                                   
        assert_equal(self.dataframe['float_col_with_int_val'].dtype, float)                  
        out_df = dataframe_column_to_str(
                             self.dataframe, 'float_col_with_int_val',                       
                             inplace=False, return_col=False)                   
        assert_equal(type(out_df), pd.DataFrame)                                
        assert_equal(out_df['float_col_with_int_val'].dtype, object)                         
        assert_equal(self.dataframe['float_col_with_int_val'].dtype, float)                  
        assert_equal(sum(pd.isnull(self.dataframe['float_col_with_int_val'])),               
                     sum(pd.isnull(out_df['float_col_with_int_val']))) 
        for idx, row in self.dataframe.iterrows():
            if pd.isnull(row['float_col_with_int_val']): 
                continue
            assert_equal(str(int(row['float_col_with_int_val'])),
                         out_df.loc[idx]['float_col_with_int_val'])

    def test_str_col_with_inplace(self):                                      
        assert_equal(self.dataframe['str_col'].dtype, object)                  
        nan_cnt_before = sum(pd.isnull(self.dataframe['str_col']))            
        flag = dataframe_column_to_str(self.dataframe, 'str_col',                         
                                       inplace=True, return_col=False)                      
        assert_equal(flag, True)                                                
        assert_equal(self.dataframe['str_col'].dtype, object)                 
        nan_cnt_after = sum(pd.isnull(self.dataframe['str_col']))             
        assert_equal(nan_cnt_before, nan_cnt_after)                             
                                                                                
    def test_str_col_with_return_col(self):                                   
        assert_equal(self.dataframe['str_col'].dtype, object)                  
        nan_cnt_before = sum(pd.isnull(self.dataframe['str_col']))            
        out_series = dataframe_column_to_str(self.dataframe, 'str_col',                   
                                             inplace=False, return_col=True)                
        assert_equal(type(out_series), pd.Series)                               
        assert_equal(out_series.dtype, object)                                  
        assert_equal(self.dataframe['str_col'].dtype, object)                  
        nan_cnt_after = sum(pd.isnull(out_series))                              
        assert_equal(nan_cnt_before, nan_cnt_after) 

    def test_int_col_with_inplace(self):                                        
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        flag = dataframe_column_to_str(self.dataframe, 'int_col',                           
                                       inplace=True, return_col=False)                      
        assert_equal(flag, True)                                                
        assert_equal(self.dataframe['int_col'].dtype, object)                   
        assert_equal(sum(pd.isnull(self.dataframe['int_col'])), 0)                             
                                                                                
    def test_int_col_with_return_col(self):                                     
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        out_series = dataframe_column_to_str(self.dataframe, 'int_col',                     
                                             inplace=False, return_col=True)                
        assert_equal(type(out_series), pd.Series)                               
        assert_equal(out_series.dtype, object)                                  
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        assert_equal(sum(pd.isnull(out_series)), 0) 

    def test_float_col_with_inplace(self):                                                   
        assert_equal(self.dataframe['float_col'].dtype, float)
        nan_cnt_before = sum(pd.isnull(self.dataframe['float_col']))                  
        flag = dataframe_column_to_str(self.dataframe, 'float_col',                       
                                       inplace=True, return_col=False)                   
        assert_equal(flag, True)                                
        assert_equal(self.dataframe['float_col'].dtype, object)                         
        nan_cnt_after = sum(pd.isnull(self.dataframe['float_col']))
        assert_equal(nan_cnt_before, nan_cnt_after)

    def test_float_col_with_return_col(self):                                      
        assert_equal(self.dataframe['float_col'].dtype, float)                  
        nan_cnt_before = sum(pd.isnull(self.dataframe['float_col']))            
        out_series = dataframe_column_to_str(self.dataframe, 'float_col',                         
                                             inplace=False, return_col=True)                      
        assert_equal(type(out_series), pd.Series)                                                
        assert_equal(out_series.dtype, object)
        assert_equal(self.dataframe['float_col'].dtype, float)                                   
        nan_cnt_after = sum(pd.isnull(out_series))             
        assert_equal(nan_cnt_before, nan_cnt_after)

    def test_nan_col_with_inplace(self):                                      
        assert_equal(self.dataframe['nan_col'].dtype, float)                  
        nan_cnt_before = sum(pd.isnull(self.dataframe['nan_col']))            
        flag = dataframe_column_to_str(self.dataframe, 'nan_col',             
                                       inplace=True, return_col=False)          
        assert_equal(flag, True)                                                
        assert_equal(self.dataframe['nan_col'].dtype, object)                 
        nan_cnt_after = sum(pd.isnull(self.dataframe['nan_col']))             
        assert_equal(nan_cnt_before, nan_cnt_after)   

    @raises(AssertionError)
    def test_invalid_dataframe(self):
        dataframe_column_to_str([], 'test_col')

    @raises(AssertionError)                                                     
    def test_invalid_col_name(self):                                           
        dataframe_column_to_str(self.dataframe, 'invalid_col')

    @raises(AssertionError)                                                     
    def test_invalid_inplace_flag(self):                                            
        dataframe_column_to_str(self.dataframe, 'str_col', inplace=None)

    @raises(AssertionError)                                                     
    def test_invalid_return_col_flag(self):                                        
        dataframe_column_to_str(self.dataframe, 'str_col', 
                                inplace=True, return_col=None)

    @raises(AssertionError)                                                     
    def test_invalid_flag_combination(self):                                        
        dataframe_column_to_str(self.dataframe, 'str_col', 
                                inplace=True, return_col=True)  

class SeriesToStrTestCases(unittest.TestCase):                         
    def setUp(self):                                                            
        self.float_col = pd.Series(np.random.randn(10)).append(                   
            pd.Series([np.NaN for _ in range(10)], index=range(10, 20)))     
        self.float_col_with_int_val = pd.Series(                                     
                                     np.random.randint(1, 100, 10)).append(  
            pd.Series([np.NaN for _ in range(10)], index=range(10, 20)))     
        self.str_col = pd.Series([random.choice(string.ascii_lowercase)              
                      for _ in range(10)]).append(                              
            pd.Series([np.NaN for _ in range(10)], index=range(10, 20)))     
        self.int_col = pd.Series(np.random.randint(1, 100, 20))                   
        self.nan_col = pd.Series([np.NaN for _ in range(20)])
                                   
    def test_str_col(self):                                                     
        assert_equal(self.str_col.dtype, object)                   
        out_series = series_to_str(self.str_col, inplace=False)       
        assert_equal(type(out_series), pd.Series)                                
        assert_equal(out_series.dtype, object)                           
        assert_equal(self.str_col.dtype, object)                   
        assert_equal(sum(pd.isnull(self.str_col)),                 
                     sum(pd.isnull(out_series)))                         
                                                                                
    def test_int_col(self):                                                    
        assert_equal(self.int_col.dtype, int)                                
        out_series = series_to_str(self.int_col, inplace=False)                     
        assert_equal(type(out_series), pd.Series)                               
        assert_equal(out_series.dtype, object)                                  
        assert_equal(self.int_col.dtype, int)                                
        assert_equal(sum(pd.isnull(out_series)), 0)   
                                                                                
    def test_float_col(self):                                                   
        assert_equal(self.float_col.dtype, float)                                
        out_series = series_to_str(self.float_col, inplace=False)                     
        assert_equal(type(out_series), pd.Series)                               
        assert_equal(out_series.dtype, object)                                  
        assert_equal(self.float_col.dtype, float)                                
        assert_equal(sum(pd.isnull(self.float_col)),                              
                     sum(pd.isnull(out_series)))  

    def test_float_col_with_int_val(self):
        assert_equal(self.float_col_with_int_val.dtype, float)                                
        out_series = series_to_str(self.float_col_with_int_val, inplace=False)                     
        assert_equal(type(out_series), pd.Series)                               
        assert_equal(out_series.dtype, object)                                  
        assert_equal(self.float_col_with_int_val.dtype, float)                                
        assert_equal(sum(pd.isnull(self.float_col_with_int_val)),                              
                     sum(pd.isnull(out_series)))                                        
        for idx, val in self.float_col_with_int_val.iteritems():                              
            if pd.isnull(val):                        
                continue                                                        
            assert_equal(str(int(val)), out_series.loc[idx])              
                                                                                
    def test_str_col_with_inplace(self):                                        
        assert_equal(self.str_col.dtype, object)                   
        nan_cnt_before = sum(pd.isnull(self.str_col))              
        flag = series_to_str(self.str_col, inplace=True)          
        assert_equal(flag, True)                                                
        assert_equal(self.str_col.dtype, object)                   
        nan_cnt_after = sum(pd.isnull(self.str_col))               
        assert_equal(nan_cnt_before, nan_cnt_after)

    def test_int_col_with_inplace(self):
        assert_equal(self.int_col.dtype, int)                                
        flag = series_to_str(self.int_col, inplace=True)                           
        assert_equal(flag, True)                                                
        assert_equal(self.int_col.dtype, object)                                
        assert_equal(sum(pd.isnull(self.int_col)), 0)         
                                        
    def test_float_col_with_inplace(self):                                      
        assert_equal(self.float_col.dtype, float)                                
        nan_cnt_before = sum(pd.isnull(self.float_col))                           
        flag = series_to_str(self.float_col, inplace=True)                           
        assert_equal(flag, True)                                                
        assert_equal(self.float_col.dtype, object)                                
        nan_cnt_after = sum(pd.isnull(self.float_col))                            
        assert_equal(nan_cnt_before, nan_cnt_after)

    # test the case with a series containing only NaN values. In this case,
    # inplace flag will be ignored.
    def test_nan_col_with_inplace(self):
        assert_equal(self.nan_col.dtype, float)                               
        nan_cnt_before = sum(pd.isnull(self.nan_col))                         
        out_series = series_to_str(self.nan_col, inplace=True)                      
        assert_equal(out_series.dtype, object)                                                
        assert_equal(self.nan_col.dtype, float)                              
        nan_cnt_after = sum(pd.isnull(out_series))                          
        assert_equal(nan_cnt_before, nan_cnt_after)

    def test_empty_series_with_inplace(self):
        empty_series = pd.Series(dtype=int)                                        
        assert_equal(empty_series.dtype, int)                                 
        out_series = series_to_str(empty_series, inplace=True)                  
        assert_equal(out_series.dtype, object)                                  
        assert_equal(empty_series.dtype, int)                                 
        assert_equal(len(out_series), 0) 
              
    @raises(AssertionError)                                                     
    def test_invalid_series(self):                                           
        series_to_str([])                                 
                                                                                
    @raises(AssertionError)                                                     
    def test_invalid_inplace_flag(self):                                        
        series_to_str(self.int_col, inplace=None)                   
