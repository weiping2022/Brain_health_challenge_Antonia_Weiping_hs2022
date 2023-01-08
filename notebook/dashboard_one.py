#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# check if one of the keyword is in a string
def key_in_str(key_list,str):
    if any(word in str for word in key_list):
        new_val = 1
    else: new_val = 0
    return new_val

# generate a new column with value: 1 if a row includes one of the keywords, 0 if not includes any
def new_col_with_key(df,key_list,new_col):
    com_col = ['Phase','RID','VISCODE2']
    df = df.astype('string')
    df['new'] = df[df.columns].apply(lambda x: '_'.join(x.dropna()), axis=1)
    L = []
    for str in df['new']:
        L.append(key_in_str(key_list,str))
    df[new_col] = L
    return df[com_col + [new_col]]

# drop the row where strings in a column cant be converted to number, and convert the rest to integer
def drop_char(df,col): 
    dff = df.copy()
    col_lst = dff[col].tolist() # extract the column to be a list
    index_lst = []  # index_l to store the index which should be droped
    for i in range(len(col_lst)):  
        if pd.notna(col_lst[i]): 
            str_ = col_lst[i]
            try:
                int(str_)
            except (RuntimeError, TypeError, NameError,ValueError):
                index_lst.append(i)  
    dff = dff.drop(index_lst)
    
    lst = dff[col].to_list()
    int_lst = []
    for i in range(len(lst)):
        to_int = int(lst[i])
        int_lst.append(to_int)
    dff[col] = int_lst
    return dff.reset_index()

# replace the strings in a column which cant be converted to float number, and convert the rest to float
def char_float_na(df,col): 
    dff = df.copy()
    col_lst = dff[col].tolist() # extract the column to be a list
    index_lst = []  # index_l to store the index which should be droped
    for i in range(len(col_lst)):  
        if pd.notna(col_lst[i]): 
            str_ = col_lst[i]
            try:
                float(str_)
            except (RuntimeError, TypeError, NameError,ValueError):
                dff=dff.replace({str_:np.NaN})
    return dff