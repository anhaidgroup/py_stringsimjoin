
# coding: utf-8

# This quickstart guide explains how to join two tables A and B using edit distance measure. First, you need to import the required packages as follows (if you have installed **py_stringsimjoin** it will automatically install the dependencies **py_stringmatching** and **pandas**):

# In[20]:


# Import libraries
import py_stringsimjoin as ssj
import py_stringmatching as sm
import pandas as pd
import os, sys


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

output_pairs = ssj.edit_distance_join_disk(A, B, 'ID', 'ID', ' name', 'title', 50,
                                      10000,n_jobs =4,l_out_attrs=[' name',' year',' director',' writers',' actors '],
                                       r_out_attrs=['title','year','director(s)','writer(s)','actor(s)'])



# In[30]:


#len(output_pairs)


# In[31]:


# examine the output pairs


# # Handling missing values

# By default, pairs with missing values are not included
# in the output. This is because a string with a missing value
# can potentially match with all strings in the other table and 
# hence the number of output pairs can become huge. If you want 
# to include pairs with missing value in the output, you need to 
# set the **allow_missing** flag to True, as shown below:

# In[32]:


#output_pairs = ssj.edit_distance_join(A, B, 'A.id', 'B.id', 'A.name', 'B.name', 5, allow_missing=True, 
 #                                     l_out_attrs=['A.name'], r_out_attrs=['B.name'])


# In[33]:


#print(output_pairs)


# # Enabling parallel processing

# If you have multiple cores which you want to exploit for performing the 
# join, you need to use the **n_jobs** option. If n_jobs is -1, all CPUs 
# are used. If 1 is given, no parallel computing code is used at all, 
# which is useful for debugging and is the default option. For n_jobs below 
# -1, (n_cpus + 1 + n_jobs) are used (where n_cpus is the total number of 
# CPUs in the machine). Thus for n_jobs = -2, all CPUs but one are used. If 
# (n_cpus + 1 + n_jobs) becomes less than 1, then no parallel computing code 
# will be used (i.e., equivalent to the default).
# 
# The following command exploits all the cores available to perform the join:

# In[34]:


#output_pairs = ssj.edit_distance_join(A, B, 'A.id', 'B.id', 'A.name', 'B.name', 5, 
#                                      l_out_attrs=['A.name'], r_out_attrs=['B.name'], n_jobs=-1)


# In[35]:


#len(output_pairs)


# You need to set n_jobs to 1 when you are debugging or you do not want 
# to use any parallel computing code. If you want to execute the join as 
# fast as possible, you need to set n_jobs to -1 which will exploit all 
# the CPUs in your machine. In case there are other concurrent processes 
# running in your machine and you do not want to halt them, then you may 
# need to set n_jobs to a value below -1.

# # Performing join on numeric attributes

# The join method expects the join attributes to be of string type. 
# If you need to perform the join over numeric attributes, then you need 
# to first convert the attributes to string type and then perform the join.
# For example, if you need to join 'A.zipcode' in table A with 'B.zipcode' in
# table B, you need to first convert the attributes to string type using 
# the following command:

# In[36]:


#ssj.dataframe_column_to_str(A, 'A.zipcode', inplace=True)
#ssj.dataframe_column_to_str(B, 'B.zipcode', inplace=True)


# Note that the above command preserves the NaN values while converting the numeric column to string type. Next, you can perform the join as shown below:

# In[37]:


#output_pairs = ssj.edit_distance_join(A, B, 'A.id', 'B.id', 'A.zipcode', 'B.zipcode', 1, 
 #                                     l_out_attrs=['A.zipcode'], r_out_attrs=['B.zipcode'])
#print(output_pairs)


# # Additional options

# You can find all the options available for the edit distance 
# join function using the **help** command as shown below:

# In[ ]:


#help(ssj.edit_distance_join)


# # More information

# Similar to edit distance measure, you can use the package to perform 
# join using other measures such as cosine, Dice, Jaccard, overlap and 
# overlap coefficient. For measures such as TF-IDF which are not 
# directly supported, you can perform the join using the filters provided 
# in the package. To know more about other join methods as well as how to 
# use filters, refer to the how-to guide (available from the 
# [package homepage](https://sites.google.com/site/anhaidgroup/projects/py_stringsimjoin)).
