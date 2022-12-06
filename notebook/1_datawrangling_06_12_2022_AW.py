#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
from dashboard_one import *
warnings.filterwarnings("ignore")


# In[2]:


com_col = ['Phase','RID','VISCODE2']   # common columns


# In[3]:


# check if one of the keyword is in a string
def key_in_str(key_list,str):
    if any(word in str for word in key_list):
        new_val = 1
    else: new_val = 0
    return new_val

# generate a new column with value: 1 if a row includes one of the keywords, 0 if not includes
def new_col_with_key(df,key_list,new_col):
    #df = pd.read_csv(file,sep=';')
    df = df.astype('string')
    df['new'] = df[df.columns].apply(lambda x: '_'.join(x.dropna()), axis=1)
    L = []
    for str in df['new']:
        L.append(key_in_str(key_list,str))
    df[new_col] = L
    return df[com_col + [new_col]]

def drop_char(df,col): # drop the row where strings in a column cant be converted to number, and convert the rest to integer
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

def char_float_na(df,col): # replace the strings in a column which cant be converted to float number, and convert the rest to float
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


# ### 1 NPI 

# #### NPI.csv 

# In[4]:


npi_1 = pd.read_csv('NPI.csv').dropna(subset = ['RID'])
pd.set_option('display.max_columns', None)
npi_1.head(2)


# select variables

# In[5]:


npi_1 = pd.concat([npi_1[com_col], npi_1.filter(regex='NPIK')], axis=1)
npi_1 = drop_char(npi_1,'RID').reset_index(drop=True)  # drop the rows where RID could not be converted to integer
npi_1 = npi_1.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID']).drop(['index'],axis=1)             # drop the duplicates and sort 
npi_1   #6756 rows


# In[6]:


npi_1.groupby(['RID']).count()


# In[7]:


npi_1[npi_1['RID']==8]   # check some example, each patient could have more than 1 not-NA records, so should use RID-VISCODE as unique features 


# #### NPIQ.csv

# In[8]:


npi_2 = pd.read_csv('NPIQ.csv').dropna(subset = ['RID'])
pd.set_option('display.max_columns',None)
npi_2.head(2)


# In[9]:


npi_2 = pd.concat([npi_2[com_col], npi_2.filter(regex='NPIK')], axis=1)
npi_2 = drop_char(npi_2,'RID').reset_index(drop=True)  #drop the rows where RID which cant be converted to integer
npi_2 = npi_2.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID']).drop(['index'],axis=1)             # drop the duplicates and sort 
npi_2   # 7010 rows


# In[10]:


npi_merge = pd.merge(npi_1,npi_2, how = 'outer', on = com_col)   # join two dataframes
npi_merge


# Missing values are marked as '-4', '-1' or NaN. 
# 
# However for NPI test, if a row contains only 0 and NaN, or in another word, if a row doesn't contain any values which are >= 1, then we should delete this row.

# In[11]:


npi_merge = npi_merge.set_index(com_col)
npi_merge.head(15)


# In[12]:


npi_boolean = (npi_merge >= 1)    # convert npi to boolean data, where True means value >=1, else False.
#npi_boolean


# In[31]:


npi_new = pd.DataFrame(npi_boolean.any(axis='columns')) # if a row has any True, will be labeled as True. Else (if all False)--> False.
npi_new.columns = ['whole_row']  # rename column
npi_new = npi_new.reset_index()  # flatten the dataframe
npi_new = npi_new[npi_new['whole_row'] == True]  # keep the rows which is true
npi_new


# In[54]:


npi_short = pd.merge(npi_new, npi_merge, on=com_col).drop(['whole_row'],axis=1)
npi_short = npi_short[npi_short['NPIK_y']!=2]
npi_short


# check if the column NPIK_x and NPIK_y could be merged as one column 'NPIK'

# In[55]:


npi_short.loc[npi_short['NPIK_x']==1,'NPIK']=1
npi_short.loc[npi_short['NPIK_y']==1,'NPIK']=1
npi_short = npi_short.drop(['NPIK_x','NPIK_y','NPIK'],axis=1)
npi_short


# In[56]:


npi_short = npi_short.replace({-4:np.nan,-1:np.nan})  # replace all -4 and -1 as nan
for value in npi_short.iloc[:,3:11]:   # replace the values in some columns as nan 
    if value!=0 and value!=1:
        value = np.nan
npi_short.to_csv('npi_short.csv')
npi_short


# ### Files for medical history 
# there are three files under assessment diagnosis: ADSXLIST,BLCHANGE,MODHACH. 
# 
# four files under medical history: MEDHIST;INITHEALTH;RECMHIST;RECBLLOG
# 
# We use these files to find out medical history of OSA,insomnia
# 
# It is not sure if they are same. Therefore all three will be checked.

# In[61]:


adsxlist = pd.read_csv('ADSXLIST.csv',sep=';').dropna(subset = ['RID'])  # drop the rows where RID is not available
blchange = pd.read_csv('BLCHANGE.csv',sep = ';').dropna(subset = ['RID'])
ass_diag_merge = pd.concat([adsxlist,blchange]).reset_index(drop=True)
ass_diag_merge = drop_char(ass_diag_merge,'RID').reset_index(drop=True).drop(['index'],axis=1)  #drop the rows where RID cant be converted to integer
ass_diag_merge
medhist = pd.read_csv('MEDHIST.csv',sep=';').dropna(subset = ['RID'])
inithealth = pd.read_csv('INITHEALTH.csv',sep = ';').dropna(subset = ['RID'])
recmhist = pd.read_csv('RECMHIST.csv',sep = ';').dropna(subset = ['RID'])
recbllog = pd.read_csv('RECBLLOG.csv',sep = ';').dropna(subset = ['RID'])
blscheck = pd.read_csv('BLSCHECK.csv',sep=';')
med_hist_merge = pd.concat([medhist,inithealth,recmhist,recbllog,blscheck]).reset_index(drop=True) # connect all rows and reset index
med_hist_merge = drop_char(med_hist_merge,'RID').reset_index(drop=True).drop(['index'],axis=1)  #drop the RID which cant be converted to integer
med_hist_merge


# In[65]:


adsxlist = pd.read_csv('ADSXLIST.csv',sep=';').dropna(subset = ['RID'])  # drop the rows where RID is not available
blchange = pd.read_csv('BLCHANGE.csv',sep = ';').dropna(subset = ['RID'])
medhist = pd.read_csv('MEDHIST.csv',sep=';').dropna(subset = ['RID'])
inithealth = pd.read_csv('INITHEALTH.csv',sep = ';').dropna(subset = ['RID'])
recmhist = pd.read_csv('RECMHIST.csv',sep = ';').dropna(subset = ['RID'])
recbllog = pd.read_csv('RECBLLOG.csv',sep = ';').dropna(subset = ['RID'])
blscheck = pd.read_csv('BLSCHECK.csv',sep=';')
OSA_files = pd.concat([medhist,inithealth,recmhist,recbllog,blscheck,adsxlist,blchange]).reset_index(drop=True) 
OSA_files = drop_char(OSA_files,'RID').reset_index(drop=True).drop(['index'],axis=1)  #drop the RID which cant be converted to integer
OSA_files


# In[ ]:





# In[60]:


key_osa = ['apnea','sleep disordered breathing','SDB','OSA']   


# ### OSA
# Till now,  sleep apnea can not be cured completely

# In[66]:


osa = new_col_with_key(OSA_files,key_osa,'OSA')    # extract key words from assessment diagnosis files
osa= drop_char(osa,'RID').reset_index(drop=True)   #drop the RID which could not be converted to integer
osa= osa.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates
osa.head(3)


# In[29]:


osa.info()


# In[30]:


osa[osa['OSA']==1].head(2)   # both osa_1 and osa_2 contain positive OSA -> the files we used are OSA-relevant


# #### Question: should each patient have only one OSA label? 

# check some patients
# 
# i.e.RID=999: 17 records/visit, but only one positive record (v06,some middle time point of the whole study),since sleep spnea is not curable, especially not in that short time.
# 
# RID=205: 17 records/visit, but only one positive record (m06, again some middle time point of the whole study)
# 
# So, each patient should have only one OSA label.

# In[67]:


osa_short = osa.groupby(['RID']).sum().reset_index().drop(['index'],axis=1)
osa_short.loc[(osa_short['OSA'] >= 1), 'OSA'] = 1
osa_short.loc[(osa_short['OSA'] == 0), 'OSA'] = 0
osa_short.to_csv('osa_short.csv')
osa_short.head(3)


# ### Insomnia
# Insomnia can be fully cured. 75% of individuals with acute insomnia were able to make a full recovery after about 12 months.

# In[70]:


insomnia_files = pd.concat([medhist,inithealth,recmhist,recbllog,blchange]).reset_index(drop=True)  #keyword searching 

insomnia_1 = new_col_with_key(insomnia_files,'insomnia','insomnia')   # extract key words from assessment diagnosis 2 files, because the ADSXLIST also has the insomnia col
insomnia_1 = insomnia_1[com_col + ['insomnia']]

insomnia_2 = adsxlist[com_col + ['AXINSOMN']] 
insomnia_2['insomnia']= adsxlist['AXINSOMN'] - 1 #take the column 'AXINSOMN' and convert 1->0, 2->1
insomnia_2 = insomnia_2.drop(['AXINSOMN'],axis=1)

insomnia_3 = pd.read_csv('BLSCHECK.csv',sep=';')[com_col + ['BCINSOMN']]
insomnia_3['insomnia'] = insomnia_3['BCINSOMN'] - 1 
insomnia_3 = insomnia_3.drop(['BCINSOMN'],axis=1)

insomnia_merge = pd.concat([insomnia_1,insomnia_2,insomnia_3])           # bind all the rows
insomnia_merge = drop_char(insomnia_merge,'RID').reset_index(drop=True)   #drop the RID which could not be converted to integer
insomnia_merge = insomnia_merge.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates
def rename_value(value):
    if value not in [0,1]:
        return np.nan
    else:
        return value
insomnia_merge['insomnia'] = insomnia_merge['insomnia'].apply(rename_value)#.drop(['index'],axis=1)   # replace the values that are not 1 or 0 to Nan
insomnia_merge = insomnia_merge.drop(['index'],axis=1).drop_duplicates().reset_index(drop=True)
insomnia_merge


# #### check some samples.
# RID=108: all records are insomnia positive.
# RID=3: all records are insomnia positive (bl showed up two times, once = 0, once 1).Many samples have similiar situation.
# 
# So, the strategie is, for the records with same RID and VISCODE, we should use the sum of the records, if sum>= 1,insomnia should be set as 1.
# 
# for the insomnia data, we need both RID and VISCODE features to link to other data.

# In[71]:


insomnia_short = insomnia_merge.groupby(['RID','VISCODE','Phase']).sum().reset_index()  # groupby the rows with same 'RID and VISCODE', and sum
insomnia_short.loc[(insomnia_short['insomnia'] >= 1), 'insomnia'] = 1 
insomnia_short


# In[72]:


insomnia_short.groupby('insomnia').count()


# In[73]:


insomnia_short.to_csv('insomnia_short.csv')
insomnia_short.head(3)


# ### ADNIMERGE  
# #### there are many interesting features, i.e. the brain volume? 
# ecog: everyday cognitive

# In[84]:


adnimerge = pd.read_csv('ADNIMERGE.csv',sep=';')
adnimerge.head(2)


# ### dataframe 1: extract the brain volumes and ABETA TAU PTAU data, and calculate the ratio to the baseline
# add new features, 'duration_in_days' indicates the days between the current exam date and the baseline
# 
# ICV, also known as TIV (Total Intracranial Volume), is the volume of the cranial cavity as taken from a 3D T1 MRI, as outlined by the supratentorial dura matter, or cerebral contour when dura is not clearly detectable.

# In[85]:


adnimerge.columns   


# In[150]:


adni_new = adnimerge[['RID','VISCODE', 'PTID','AGE','PTGENDER','PTEDUCAT','ORIGPROT','EXAMDATE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'Ventricles_bl', 'Hippocampus_bl',
 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'ABETA','TAU','PTAU', 'ABETA_bl', 'TAU_bl', 'PTAU_bl']].reset_index().drop(['index'],axis=1)
adni_new = adni_new.replace({-4:np.nan,-1:np.nan,'-4':np.nan,'-1':np.nan})
adni_new['ratio_Ventricles_bl'] = adni_new['Ventricles']/adni_new['Ventricles_bl']
adni_new['ratio_Hippocampus_bl'] = adni_new['Hippocampus']/adni_new['Hippocampus_bl']
adni_new['ratio_WholeBrain_bl'] = adni_new['WholeBrain']/adni_new['WholeBrain_bl']
adni_new['ratio_Entorhinal_bl'] = adni_new['Entorhinal']/adni_new['Entorhinal_bl']
adni_new['ratio_Fusiform_bl'] = adni_new['Fusiform']/adni_new['Fusiform_bl']
adni_new['ratio_ICV_bl'] = adni_new['ICV']/adni_new['ICV_bl']
# There are some strings in the following 6 columns, I will replace strings to NA, and convert others to float number.
lst = ['ABETA','TAU', 'PTAU','ABETA_bl', 'TAU_bl', 'PTAU_bl']
for i in range(len(lst)):
    col = lst[i]
    adni_new =  char_float_na(adni_new,col).reset_index().drop(['index'],axis=1)
    adni_new[col] = adni_new[col].astype(float)
adni_new['ratio_ABETA_bl'] = adni_new['ABETA']/adni_new['ABETA_bl']
adni_new['ratio_TAU_bl'] = adni_new['TAU']/adni_new['TAU_bl']
adni_new['ratio_PTAU_bl'] = adni_new['PTAU']/adni_new['PTAU_bl']
adni_new = adni_new.rename({'ORIGPROT': 'Phase'}, axis=1)
adni_new.head(3)


# In[151]:


adni_new[adni_new['ratio_WholeBrain_bl']<1]


# In[152]:


adni_new[adni_new['ratio_WholeBrain_bl']>1.03]


# In[153]:


sns.histplot(data=adni_new, x="ratio_WholeBrain_bl",bins=50)


# In[154]:


adni_exam_bl.groupby(['RID','Phase']).count()


# In[155]:


adni_exam_bl


# In[156]:


'''df = adni_new.copy()
df.bl = df.groupby('RID').EXAMDATE
df = df.dropna(subset = 'PTDOBYY')'''

adni_exam_bl = adni_new[adni_new['VISCODE']=='bl']#[com_col + ['EXAMDATE']]
adni_exam_bl = adni_exam_bl.rename({'EXAMDATE': 'EXAMDATE_bl'}, axis=1)[['RID','Phase','EXAMDATE_bl']]
# left merge the EXAMDATE_bl to adni_new dataframe on 'RID' and 'Phase', because each RID has many baselines
adni_new = pd.merge(adni_new,adni_exam_bl, how = 'left', on=['RID','Phase'] )

# convert the date into datetime form
from datetime import datetime as dt
adni_new[['EXAMDATE','EXAMDATE_bl']] = adni_new[['EXAMDATE','EXAMDATE_bl']].apply(pd.to_datetime)
#generate new feature: duration in days between the baseline date to the current date
adni_new['duration_in_days'] = (adni_new['EXAMDATE'] - adni_new['EXAMDATE_bl']).astype('timedelta64[D]')
adni_new.head(3)


# In[157]:


# add new features: the reduction of brain volume per year
cat_lst = ['ratio_Ventricles_bl','ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
       'ratio_Fusiform_bl', 'ratio_ICV_bl', 'ratio_ABETA_bl', 'ratio_TAU_bl',
       'ratio_PTAU_bl']
new_cat_lst = ['Ventricles_reduction_per_year','Hippocampus_reduction_per_year','wholebrain_reduction_per_year','Entorhinal_reduction_per_year',
               'Fusiform_reduction_per_year','ICV_reduction_per_year','ABETA_reduction_per_year','TAU_reduction_per_year','PTAU_reduction_per_year']
for i in range(len(cat_lst)):
    new_col = new_cat_lst[i]
    old_col = cat_lst[i]
    adni_new[new_col] = (1 - adni_new[old_col])/adni_new['duration_in_days']*365
adni_new


# In[158]:


adni_not_bl = adni_new[adni_new['VISCODE']!= 'bl']


# In[159]:


fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18,20))
axes = axes.ravel()  # array to 1D
for col, ax in zip(cat_lst, axes):
    sns.histplot(data=adni_not_bl, x=col, ax=ax,bins=100)
    ax.set(title=f'Distribution of {col}', xlabel=None)   
fig.tight_layout()
plt.show()


# In[160]:


fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(18,20))
axes = axes.ravel()  # array to 1D
for col, ax in zip(new_cat_lst, axes):
    sns.histplot(data=adni_not_bl, x=col, ax=ax,bins=100)
    ax.set(title=f'Distribution of {col}', xlabel=None)   
fig.tight_layout()
plt.show()


# In[161]:


adni_new.to_csv('adnimerge_new.csv')


# ### DXSUM_PDXCONV_ADNIALL: Diagnostic information

# In[162]:


dxsum = pd.read_csv('DXSUM_PDXCONV_ADNIALL.csv',sep=';')
dxsum.head(2)


# In[163]:


diag_short = dxsum[['Phase', 'RID', 'VISCODE','PTID','DXCHANGE', 'DXCURREN', 'DIAGNOSIS']].drop_duplicates().reset_index(drop=True).sort_values(by= ['RID']) 
diag_short = drop_char(diag_short,'RID').reset_index(drop=True).drop(['index'],axis=1)  #drop the row where RID cant be converted to integer
diag_short.to_csv('diag_adniall_short.csv')
diag_short.head(2)


# In[164]:


diag_short.info()


# ### join dataframes

# In[165]:


files = [insomnia_short,npi_short]
com_col2 = ['Phase', 'RID', 'VISCODE','PTID']
main_file = pd.merge(adni_new, diag_short,how='outer', on=com_col2)
for file in files:
        additional_data = file
        main_file = pd.merge(main_file, additional_data,
                             how='outer', on=com_col)

        
main_file = pd.merge(main_file, osa_short,how='left', on='RID')
main_file


# In[82]:


main_file = main_file.replace({-4:np.nan,-1:np.nan,'-4':np.nan,'-1':np.nan})
main_file = main_file.sort_values(by = ['RID'])
# Order  by VISCODE, Phases
main_file['Phase'] = main_file['Phase'].astype('category')
main_file['Phase'] = main_file['Phase'].cat.set_categories(['ADNI1', 'ADNIGO', 'ADNI2', 'ADNI3'], ordered=True)

main_file['VISCODE'] = main_file['VISCODE'].astype('category')
main_file['VISCODE'] = main_file['VISCODE'].cat.set_categories(['sc', 'bl', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48',
                                                  'm54', 'uns1', 'f', 'nv', 'scmri', 'm03', 'm60', 'm66', 'm72', 'm78', 'v01', 'v02',
                                                  'v03', 'v04', 'v05', 'v06', 'v07', 'v11', 'v12', 'v21', 'v22', 'v31',
                                                  'v32', 'v41', 'v42', 'v51', 'v52', 'tau', 'reg', 'init', 'y1', 'y2', 
                                                  'y3', 'y4', 'y5', 'y6'], ordered=True)
main_file = main_file.sort_values(by = ['RID','Phase','VISCODE'])
main_file


# In[168]:


main_file.to_csv('main_file_1206.csv')


# ###   06_12_2022

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### complete the 'DXCHANGE'

# - ADNI1: DXCURREN 1=NL; 2=MCI; 3=AD
# - ADNIGO/2: DXCHANGE
# - 1=Stable: NL to NL;
# - 2=Stable: MCI to MCI;
# - 3=Stable: Dementia to Dementia;
# - 4=Conversion: NL to MCI;
# - 5=Conversion: MCI to Dementia; 
# - 6=Conversion: NL to Dementia;
# - 7=Reversion: MCI to NL;
# - 8=Reversion: Dementia to MCI;
# - 9=Reversion: Dementia to NL
# - ADNI3: DIAGNOSIS 1=CN; 2=MCI; 3=Dementia

# - Mild Cognitive Impairment (MCI)
# - Alzheimer's Disease (AD)
# - Significant Memory Concern (SMC)
# - Early Mild Cognitive Impairment (EMCI)
# - Late Mild Cognitive Impairment (LMCI)

# In[166]:


main_file.loc[(main_file['DIAGNOSIS'] == 1.0)|(main_file['DXCURREN']==1.0), 'DX'] = 'CN'
main_file.loc[(main_file['DIAGNOSIS'] == 2.0)|(main_file['DXCURREN']==2.0), 'DX'] = 'MCI'
main_file.loc[(main_file['DIAGNOSIS'] == 3.0)|(main_file['DXCURREN']==3.0), 'DX'] = 'AD'
main_file = main_file.drop(['DIAGNOSIS','DXCURREN'],axis=1)
main_file = main_file[main_file.VISCODE != 'f']   # drop the rows where VISCODE == 'f'
main_file


# #### Fill NA in column 'CN': For each patient, if row i is Nan, check if the row i+1 and i-1 are same, if yes, then fill row i with the same value. 

# In[84]:


dff = main_file.copy()
dff['DX'] = dff['DX'].replace(np.nan,-1)
data = pd.DataFrame()
unq_rid = dff['RID'].unique()
for j in range(len(unq_rid)):
    rid = unq_rid[j]  # select RID
    df_new = dff[dff['RID']== rid] #new dataframe only includes one patient
    length = len(df_new)
    m = df_new.copy()
    m['DX'] = -1
    df_new = df_new.append(m)

    for i in range(1,len(df_new)-length):
        if df_new.iloc[i,13] == -1:
            for m in range(1,length+1):
                if df_new.iloc[i-1,13] == df_new.iloc[i+m,13] and df_new.iloc[i-1,13]!= -1:
                    df_new.iloc[i,13] = df_new.iloc[i-1,13]
                    break

    df_new = df_new.iloc[0:length,:]            
    data = data.append(df_new)
main_file = data.copy()
main_file['DX'] = main_file['DX'].replace(-1,np.nan)
main_file = main_file.reset_index().drop(['index'],axis=1)
main_file


# in different Phases one patient has different dx baseline 

# In[86]:


# add new feature 'RID_Phase'
main_file['RID_s'] = main_file['RID'].astype(str)
main_file['RID_Phase'] = main_file[['RID_s', 'Phase']].agg('_'.join, axis=1)
main_file = main_file.drop(['RID_s'],axis=1)


# In[87]:


main_file['DX_bl']=main_file['DX_bl'].replace({'LMCI':'MCI', 'EMCI':'MCI','SMC':'CN'})
main_file['DX']=main_file['DX'].replace({'Dementia':'AD'})


# In[88]:


dff = main_file.copy()
dff[dff['Phase']=='ADNI2'].tail(50)


# In[89]:


dff = main_file.copy()
dff = dff[['RID','DX','DX_bl','RID_Phase','VISCODE']]
dff['DX_bl']=dff['DX_bl'].replace({np.nan:-1})
dff['DX']=dff['DX'].replace({np.nan:-1})
l = []
for i in range(len(dff)):
    if dff.iloc[i,1] != dff.iloc[i,2]:
        l.append(i)
a=dff.iloc[l]
m = a[(a['VISCODE']=='bl') & (a['DX']!= -1)]           # check the data where    there is conflict on baseline data   
m


# In[90]:


dff[dff['RID_Phase']=='78_ADNI1']   


# In[91]:


conflict_lst = m['RID_Phase'].to_list()
conflict_lst


# ### 1226ADNI1, 2201_ADNIGO,1154_ADNI1,995_ADNI1,739_ADNI1,332_ADNI1,190_ADNI1,78_ADNI1 should be deleted. because there are conflict on baseline data.

# In[92]:


main_file = main_file[~main_file['RID_Phase'].isin(conflict_lst)]
main_file


# In[93]:


dx_bl = main_file[['DX_bl','RID_Phase']]   # diagnosis baseline

dx_bl = dx_bl.dropna(how='any',axis=0).drop_duplicates(keep='first')
len(dx_bl) == dx_bl['RID_Phase'].nunique()   # each RID_Phase only shows up once, so could merge it to main_file


# In[94]:


main_file = main_file.drop(['DX_bl'],axis=1)


# In[95]:


main_file = main_file.merge(dx_bl,how='left',on='RID_Phase')


# In[96]:


main_file.loc[(main_file['DX_bl'] == 'CN')&(main_file['DX']=='CN'), 'DXCHANGE'] = 'CN-CN'
main_file.loc[(main_file['DX_bl'] == 'MCI')&(main_file['DX']=='MCI'), 'DXCHANGE'] = 'MCI-MCI'
main_file.loc[((main_file['DX_bl'] == 'Dementia')|(main_file['DX_bl'] == 'AD'))&((main_file['DX']=='Dementia')|(main_file['DX'] == 'AD')), 'DXCHANGE'] = 'AD-AD'
main_file.loc[(main_file['DX_bl'] == 'CN')&(main_file['DX']=='MCI'), 'DXCHANGE'] = 'CN-MCI'
main_file.loc[(main_file['DX_bl'] == 'MCI')&((main_file['DX']=='Dementia')|(main_file['DX'] == 'AD')), 'DXCHANGE'] = 'MCI-AD'
main_file.loc[(main_file['DX_bl'] == 'CN')&((main_file['DX']=='Dementia')|(main_file['DX'] == 'AD')), 'DXCHANGE'] = 'CN-AD'
main_file.loc[(main_file['DX_bl'] == 'MCI')&(main_file['DX']=='CN'), 'DXCHANGE'] = 'MCI-CN'
main_file.loc[((main_file['DX_bl'] == 'Demencia')|(main_file['DX_bl'] == 'AD'))&(main_file['DX']=='MCI'), 'DXCHANGE'] = 'AD-MCI'
main_file.loc[((main_file['DX_bl'] == 'Demencia')|(main_file['DX_bl'] == 'AD'))&(main_file['DX']=='CN'), 'DXCHANGE'] = 'AD-CN'
main_file = main_file.replace({'DXCHANGE':{1.0:'CN-CN',2.0: 'MCI-MCI',3.0:'AD-AD',4.0:'CN-MCI',5.0:'MCI-AD',6.0:'CN-AD',7.0:'MCI-CN',8.0:'AD-MCI',9.0:'AD-CN'}})
main_file['DXCHANGE'].unique()


# In[97]:


main_file.to_csv('main_file.csv')
main_file


# ## Use the last diagnosis of each patient! 
# 
# The final goal is to use current data to predict for future NOT to use current data to predict current
# Use the phase of one patient, only when he/she has at least 3 DXCHANGE records in this phase.

# In[134]:


dx = ['DX_bl','DXCHANGE','DX']


# ### Final diagnosis changes of each patient in each pahse

# ### for each patient each phase is a new start: new bl 

# In[135]:


main_1 = main_file.copy()
main_1= main_1.dropna(subset = ['DXCHANGE'])  
a = main_1.groupby('RID_Phase').count()
RID_Phase_lst = a[a['RID']>2].index  # list of selected RID_Phase
# drop the rows where RID count is less than 2, because then it is too short time to say that the unchange diagnosis is unchange
mainfile_finaldxch = main_1[main_1['RID_Phase'].isin(RID_Phase_lst)]
mainfile_finaldxch = mainfile_finaldxch[mainfile_finaldxch['VISCODE']!='sc'].reset_index().drop(['index'],axis=1)
# to get the final diagnosis changes of each RID_Phase.
final_dxch = mainfile_finaldxch[['RID_Phase','DXCHANGE']].dropna(how='any',axis=0).groupby('RID_Phase').tail(1) # keep only the final dxch
final_dxch.columns = ['RID_Phase', 'final_dxch']  # keep only the RID_Phase and final_dxch


# split into two parts, one part where patient through the whole Phase has no changes.
# 
# another part patient has changes within one phase, for this part, we will keep only the data before it changes

# In[136]:


final_dxch_unchange =  final_dxch[final_dxch['final_dxch'].isin(['CN-CN','AD-AD','MCI-MCI'])]
final_dxch_change =  final_dxch[~final_dxch['final_dxch'].isin(['CN-CN','AD-AD','MCI-MCI'])]
# merge to main_file
mainfile_final_dxch_unchange = mainfile_finaldxch.merge(final_dxch_unchange,how='right',on='RID_Phase')  
mainfile_final_dxch_change = mainfile_finaldxch.merge(final_dxch_change,how='right',on='RID_Phase')  


# In[137]:


data = pd.DataFrame()
unq_rid = mainfile_final_dxch_change['RID_Phase'].unique()
for i in range(len(unq_rid)):
    rid_ = unq_rid[i]
    df_ = mainfile_final_dxch_change[mainfile_final_dxch_change['RID_Phase']== rid_]  # new dataframe with only one patient

    for j in range(len(df_)):        
        if df_.iloc[j,19] == df_.iloc[j,-1]:   # keep only the data before the changes
            continue #break
        else: data = data.append(df_.iloc[[j]])
mainfile_final_dxch_change = data.reset_index().drop(['index'],axis=1)
mainfile_finaldxch = pd.concat([mainfile_final_dxch_change,mainfile_final_dxch_unchange])
mainfile_finaldxch


# merge brain volume data and sleep data

# In[138]:


df1 = main_file[['RID', 'Phase', 'VISCODE','PTID', 
       'NPIK1', 'NPIK2', 'NPIK3', 'NPIK4', 'NPIK5', 'NPIK6', 'NPIK7', 'NPIK8',
       'NPIK9A', 'NPIK9B', 'NPIK9C', 'NPIKTOT', 'NPIKSEV', 'insomnia','OSA','DXCHANGE']].set_index(['RID', 'Phase', 'VISCODE','PTID']).dropna(how='all',axis=0).reset_index()
df1


# In[139]:


df2 = mainfile_finaldxch[['RID', 'Phase', 'VISCODE','PTID', 
       'NPIK1', 'NPIK2', 'NPIK3', 'NPIK4', 'NPIK5', 'NPIK6', 'NPIK7', 'NPIK8',
       'NPIK9A', 'NPIK9B', 'NPIK9C', 'NPIKTOT', 'NPIKSEV', 'insomnia','OSA','final_dxch','DXCHANGE','RID_Phase']].set_index(['RID', 'Phase', 'VISCODE','PTID','RID_Phase']).dropna(how='all',axis=0).reset_index()  


# In[140]:


df4 = adni_new[['RID', 'Phase', 'VISCODE','PTID','ratio_Ventricles_bl',
       'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
       'ratio_Fusiform_bl', 'ratio_ICV_bl', 'ratio_ABETA_bl', 'ratio_TAU_bl',
       'ratio_PTAU_bl','Ventricles_reduction_per_year', 'Hippocampus_reduction_per_year',
       'wholebrain_reduction_per_year', 'Entorhinal_reduction_per_year',
       'Fusiform_reduction_per_year', 'ICV_reduction_per_year',
       'ABETA_reduction_per_year', 'TAU_reduction_per_year',
       'PTAU_reduction_per_year']].set_index(['RID', 'Phase', 'VISCODE','PTID']).dropna(how='all',axis=0).reset_index()
df4


# In[141]:


sleep_brain_dxch = pd.merge(df1,df4, how = 'outer', on = ['RID', 'Phase', 'VISCODE','PTID']) 
sleep_brain_dxch = sleep_brain_dxch[sleep_brain_dxch['VISCODE']!='bl']
sleep_brain_dxch.to_csv('sleep_brain_dxch.csv')
sleep_brain_dxch.info()


# In[142]:


sleep_brain_finaldxch = pd.merge(df2,df4, how = 'outer', on = ['RID', 'Phase', 'VISCODE','PTID']) 
sleep_brain_finaldxch = sleep_brain_finaldxch[sleep_brain_finaldxch['VISCODE']!='bl']
sleep_brain_finaldxch.to_csv('sleep_brain_finaldxch.csv')
sleep_brain_finaldxch.info()


# In[ ]:





# In[ ]:





# In[ ]:




