
# coding: utf-8

# This quickstart guide explains how to join two tables A and B using edit distance measure. First, you need to import the required packages as follows (if you have installed **py_stringsimjoin** it will automatically install the dependencies **py_stringmatching** and **pandas**):

# In[20]:


# Import libraries
import py_stringsimjoin as ssj
import py_stringmatching as sm
import pandas as pd
import os, sys
import time


# In[21]:


print('python version: ' + sys.version)
print('py_stringsimjoin version: ' + ssj.__version__)
print('py_stringmatching version: ' + sm.__version__)
print('pandas version: ' + pd.__version__)
print(sys.path)


# Joining two tables using edit distance measure typically consists of three steps:
# 1. Loading the input tables
# 2. Profiling the tables
# 3. Performing the join

# # 1. Loading the input tables

# We begin by loading the two tables. For the purpose of this guide, 
# we use the sample dataset that comes with the package. 

# In[22]:


# construct the path of the tables to be loaded. Since we are loading a 
# dataset from the package, we need to access the data from the path 
# where the package is installed. If you need to load your own data, you can directly
# provide your table path to the read_csv command.

table_A_path = os.sep.join([ssj.get_install_path(), 'datasets', 'data', 'imdb_A.csv'])
table_B_path = os.sep.join([ssj.get_install_path(), 'datasets', 'data', 'imdb_B.csv'])


# In[23]:


# Load csv files as dataframes.
A = pd.read_csv(table_A_path,error_bad_lines= False)
B = pd.read_csv(table_B_path,error_bad_lines=False)
print('Number of records in A: ' + str(len(A)))
print('Number of records in B: ' + str(len(B)))
print(A.columns.values)
print(B.columns.values)

# In[24]:




# In[25]:





# # 2. Profiling the tables

# Before performing the join, we may want to profile the tables to 
# know about the characteristics of the attributes. This can help identify:
# 
# a) unique attributes in the table which can be used as key attribute when performing 
#    the join. A key attribute is needed to uniquely identify a tuple. 
#    
# b) the number of missing values present in each attribute. This can 
#    help you in deciding the attribute on which to perform the join. 
#    For example, an attribute with a lot of missing values may not be a good 
#    join attribute. Further, based on the missing value information you 
#    need to decide on how to handle missing values when performing the join 
#    (See the section below on 'Handling missing values' to know more about
#    the options available for handling missing values when performing the join).
#    
# You can profile the attributes in a table using the following command:

# In[26]:


# profile attributes in table A
#print(ssj.profile_table_for_join(A))


# In[27]:


# profile attributes in table B
#print(ssj.profile_table_for_join(B))


# If the input tables does not contain any key attribute, then you need 
# to create a key attribute. In the current example, both the input tables
# A and B have key attributes, and hence you can proceed to the next step.
# In the case the table does not have any key attribute, you can 
# add a key attribute using the following command:

# In[28]:


#B['new_key_attr'] = range(0, len(B))
#print(B)


# For the purpose of this guide, we will now join tables A and B on 
# 'name' attribute using edit distance measure. Next, we need to decide on what 
# threshold to use for the join. For this guide, we will use a threshold of 5. 
# Specifically, the join will now find tuple pairs from A and B such that 
# the edit distance over the 'name' attributes is at most 5.

# # 3. Performing the join

# The next step is to perform the edit distance join using the following command:

# In[29]:


# find all pairs from A and B such that the edit distance
# on 'name' is at most 5.
# l_out_attrs and r_out_attrs denote the attributes from the 
# left table (A) and right table (B) that need to be included in the output.

tick = time.time()

output_pairs = ssj.edit_distance_join_disk(A, B, 'ID', 'ID', ' name', 'title', 1,
                                      100000,n_jobs =4,l_out_attrs=[' name',' year'],#' director',' writers',' actors '],
                                       r_out_attrs=['title','year'], temp_file_path = "/afs/cs.wisc.edu/u/a/j/ajain64/private/Spring_2018/indep/git/",
                                           allow_missing = True, output_file_path= "/afs/cs.wisc.edu/u/a/j/ajain64/private/Spring_2018/indep/join_output.csv")

tock = time.time()


print ("Time taken in disk " + str(tock-tick))