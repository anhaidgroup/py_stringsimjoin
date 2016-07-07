import random
import string
import unittest

from nose.tools import assert_equal, assert_list_equal, raises
import pandas as pd

from py_stringsimjoin.utils.converter import convert2str


class ConvertToStringTestCases(unittest.TestCase):
    def setUp(self):
        float_col = pd.Series(pd.np.random.randn(10)).append(
            pd.Series([pd.np.NaN for _ in range(10)], index=range(10, 20)))
        float_col_with_int_val = pd.Series(
                                     pd.np.random.randint(1, 100, 10)).append(          
            pd.Series([pd.np.NaN for _ in range(10)], index=range(10, 20)))        
        str_col = pd.Series([random.choice(string.ascii_lowercase) 
                      for _ in range(10)]).append(
            pd.Series([pd.np.NaN for _ in range(10)], index=range(10, 20)))
        int_col = pd.Series(pd.np.random.randint(1, 100, 20))                           

        self.dataframe = pd.DataFrame({'float_col': float_col,
                               'float_col_with_int_val': float_col_with_int_val,
                               'int_col': int_col,
                               'str_col': str_col})

    def test_str_col(self):
        assert_equal(self.dataframe['str_col'].dtype, object)
        out_df = convert2str(self.dataframe, 'str_col', 
                             inplace=False, return_col=False)
        assert_equal(type(out_df), pd.DataFrame)
        assert_equal(out_df['str_col'].dtype, object)
        assert_equal(self.dataframe['str_col'].dtype, object)                   
        assert_equal(sum(pd.isnull(self.dataframe['str_col'])), 
                     sum(pd.isnull(out_df['str_col'])))                     

    def test_int_col(self):                                                     
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        out_df = convert2str(self.dataframe, 'int_col',                         
                             inplace=False, return_col=False)                   
        assert_equal(type(out_df), pd.DataFrame)                                
        assert_equal(out_df['int_col'].dtype, object)
        assert_equal(self.dataframe['int_col'].dtype, int)                     
        assert_equal(sum(pd.isnull(out_df['int_col'])), 0)

    def test_float_col(self):                                                     
        assert_equal(self.dataframe['float_col'].dtype, float)                   
        out_df = convert2str(self.dataframe, 'float_col',                         
                             inplace=False, return_col=False)                   
        assert_equal(type(out_df), pd.DataFrame)                                
        assert_equal(out_df['float_col'].dtype, object)
        assert_equal(self.dataframe['float_col'].dtype, float)                                             
        assert_equal(sum(pd.isnull(self.dataframe['float_col'])),                           
                     sum(pd.isnull(out_df['float_col']))) 

    def test_float_col_with_int_val(self):                                                   
        assert_equal(self.dataframe['float_col_with_int_val'].dtype, float)                  
        out_df = convert2str(self.dataframe, 'float_col_with_int_val',                       
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
                         out_df.ix[idx]['float_col_with_int_val'])

    def test_str_col_with_inplace(self):                                      
        assert_equal(self.dataframe['str_col'].dtype, object)                  
        nan_cnt_before = sum(pd.isnull(self.dataframe['str_col']))            
        flag = convert2str(self.dataframe, 'str_col',                         
                           inplace=True, return_col=False)                      
        assert_equal(flag, True)                                                
        assert_equal(self.dataframe['str_col'].dtype, object)                 
        nan_cnt_after = sum(pd.isnull(self.dataframe['str_col']))             
        assert_equal(nan_cnt_before, nan_cnt_after)                             
                                                                                
    def test_str_col_with_return_col(self):                                   
        assert_equal(self.dataframe['str_col'].dtype, object)                  
        nan_cnt_before = sum(pd.isnull(self.dataframe['str_col']))            
        out_series = convert2str(self.dataframe, 'str_col',                   
                                 inplace=False, return_col=True)                
        assert_equal(type(out_series), pd.Series)                               
        assert_equal(out_series.dtype, object)                                  
        assert_equal(self.dataframe['str_col'].dtype, object)                  
        nan_cnt_after = sum(pd.isnull(out_series))                              
        assert_equal(nan_cnt_before, nan_cnt_after) 

    def test_int_col_with_inplace(self):                                        
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        flag = convert2str(self.dataframe, 'int_col',                           
                           inplace=True, return_col=False)                      
        assert_equal(flag, True)                                                
        assert_equal(self.dataframe['int_col'].dtype, object)                   
        assert_equal(sum(pd.isnull(self.dataframe['int_col'])), 0)                             
                                                                                
    def test_int_col_with_return_col(self):                                     
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        out_series = convert2str(self.dataframe, 'int_col',                     
                                 inplace=False, return_col=True)                
        assert_equal(type(out_series), pd.Series)                               
        assert_equal(out_series.dtype, object)                                  
        assert_equal(self.dataframe['int_col'].dtype, int)                   
        assert_equal(sum(pd.isnull(out_series)), 0) 

    def test_float_col_with_inplace(self):                                                   
        assert_equal(self.dataframe['float_col'].dtype, float)
        nan_cnt_before = sum(pd.isnull(self.dataframe['float_col']))                  
        flag = convert2str(self.dataframe, 'float_col',                       
                           inplace=True, return_col=False)                   
        assert_equal(flag, True)                                
        assert_equal(self.dataframe['float_col'].dtype, object)                         
        nan_cnt_after = sum(pd.isnull(self.dataframe['float_col']))
        assert_equal(nan_cnt_before, nan_cnt_after)

    def test_float_col_with_return_col(self):                                      
        assert_equal(self.dataframe['float_col'].dtype, float)                  
        nan_cnt_before = sum(pd.isnull(self.dataframe['float_col']))            
        out_series = convert2str(self.dataframe, 'float_col',                         
                                 inplace=False, return_col=True)                      
        assert_equal(type(out_series), pd.Series)                                                
        assert_equal(out_series.dtype, object)
        assert_equal(self.dataframe['float_col'].dtype, float)                                   
        nan_cnt_after = sum(pd.isnull(out_series))             
        assert_equal(nan_cnt_before, nan_cnt_after)

    @raises(AssertionError)
    def test_invalid_dataframe(self):
        convert2str([], 'test_col')

    @raises(AssertionError)                                                     
    def test_invalid_col_name(self):                                           
        convert2str(self.dataframe, 'invalid_col')

    @raises(AssertionError)                                                     
    def test_invalid_inplace_flag(self):                                            
        convert2str(self.dataframe, 'str_col', inplace=None)

    @raises(AssertionError)                                                     
    def test_invalid_return_col_flag(self):                                        
        convert2str(self.dataframe, 'str_col', inplace=True, return_col=None)

    @raises(AssertionError)                                                     
    def test_invalid_flag_combination(self):                                        
        convert2str(self.dataframe, 'str_col', inplace=True, return_col=True)  

