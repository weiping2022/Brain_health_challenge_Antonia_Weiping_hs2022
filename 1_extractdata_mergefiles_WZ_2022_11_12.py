#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


com_col = ['Phase','RID','VISCODE']   # common columns


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
npi_1 = npi_1.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates and sort 
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
npi_2 = npi_2.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates and sort 
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


# In[13]:


npi_new = pd.DataFrame(npi_boolean.any(axis='columns')) # if a row has any True, will be labeled as True. Else (if all False)--> False.
npi_new.columns = ['whole_row']  # rename column
npi_new = npi_new.reset_index()  # flatten the dataframe
npi_new = npi_new[npi_new['whole_row'] == True]  # keep the rows which is true
npi_new


# In[14]:


npi_short = pd.merge(npi_new, npi_merge, on=com_col).drop(['whole_row'],axis=1)
npi_short


# check if the column NPIK_x and NPIK_y could be merged as one column 'NPIK'

# In[15]:


npi_short['NPIK'] = 100
for i in range(len(npi_short)):
    if np.isnan(npi_short['NPIK_x'][i])==False and np.isnan(npi_short['NPIK_y'][i])==True:
        npi_short['NPIK'][i] = npi_short['NPIK_x'][i]
    elif np.isnan(npi_short['NPIK_x'][i])==True and np.isnan(npi_short['NPIK_y'][i])==False:
        npi_short['NPIK'][i] = npi_short['NPIK_y'][i]
    elif np.isnan(npi_short['NPIK_x'][i])==True and np.isnan(npi_short['NPIK_y'][i])==True:
        npi_short['NPIK'][i] = np.nan
    elif np.isnan(npi_short['NPIK_x'][i])==False and np.isnan(npi_short['NPIK_y'][i])==False:
        if npi_short['NPIK_x'][i] == npi_short['NPIK_y'][i]:
            npi_short['NPIK'][i] = npi_short['NPIK_x'][i]
        else: npi_short['NPIK'][i] = 5
npi_short.groupby(['NPIK']).count()  


# NPIK has the value of 1, 2(NA), 5(conflict).
# 
# I will replace the NPIK value of 2 and 5 as Nan.Afterwards, NPIK is 1 for all samples, the 0 never appeared. In this case, this variable will not help us for further analysis. So I will drop the column 'NPIK'.
# 
# except NPIKSEV, NPIKTOT, NPIK9A 9B 9C, all other NPIK columns should only have value 0 or 1 or Nan.

# In[16]:


npi_short = npi_short.drop(['NPIK_x','NPIK_y','NPIK'],axis=1)
npi_short = npi_short.replace({-4:np.nan,-1:np.nan})  # replace all -4 and -1 as nan
for value in npi_short.iloc[:,3:11]:   # replace the values in some columns as nan 
    if value!=0 and value!=1:
        value = np.nan
npi_short.to_csv('npi_short.csv')


# ### Files for medical history 
# there are three files under assessment diagnosis: ADSXLIST,BLCHANGE,MODHACH. 
# 
# four files under medical history: MEDHIST;INITHEALTH;RECMHIST;RECBLLOG
# 
# We use these files to find out medical history of hypertension,OSA,anxiety,depression,insomnia
# 
# It is not sure if they are same. Therefore all three will be checked.

# In[108]:


adnimerge


# In[17]:


adsxlist = pd.read_csv('ADSXLIST.csv',sep=';').dropna(subset = ['RID'])  # drop the rows where RID is not available
blchange = pd.read_csv('BLCHANGE.csv',sep = ';').dropna(subset = ['RID'])
modhach = pd.read_csv('MODHACH.csv',sep = ';').dropna(subset = ['RID'])
ass_diag_merge = pd.concat([adsxlist,blchange,modhach]).reset_index(drop=True)
ass_diag_merge = drop_char(ass_diag_merge,'RID').reset_index(drop=True)  #drop the rows where RID cant be converted to integer
ass_diag_merge


# In[18]:


medhist = pd.read_csv('MEDHIST.csv',sep=';').dropna(subset = ['RID'])
inithealth = pd.read_csv('INITHEALTH.csv',sep = ';').dropna(subset = ['RID'])
recmhist = pd.read_csv('RECMHIST.csv',sep = ';').dropna(subset = ['RID'])
recbllog = pd.read_csv('RECBLLOG.csv',sep = ';').dropna(subset = ['RID'])
med_hist_merge = pd.concat([medhist,inithealth,recmhist,recbllog]).reset_index(drop=True) # connect all rows and reset index
med_hist_merge = drop_char(med_hist_merge,'RID').reset_index(drop=True)  #drop the RID which cant be converted to integer
med_hist_merge


# In[19]:


key_osa = ['apnea','sleep disordered breathing','SDB','OSA']   
key_hypertension = ['hypertension','Hypertension','HTN','htn'] 


# ### 2. hypertension
# It is important to know that, we have to use the right data to search for keywords.
# 
# i.e. we should only use the file which involve the information about 'hypertension' to evaluate if the patients in this file has hypertension or not.
# 
# If we use a file which not involve the information about 'hypertension' to evaluate hypertension patients, then all the patients in this file will be labeled as 'not-have-hypertension', which is not true, their hypertension info are not available, should be labeled as 'NA'. 

# In[20]:


htn_1 = new_col_with_key(ass_diag_merge,key_hypertension,'hypertension')    # extract key words from assessment diagnosis files
htn_2 = new_col_with_key(med_hist_merge,key_hypertension,'hypertension')   # extract key words from medical history files
htn_3 = modhach[com_col + ['HMHYPERT']]                     # from column'HMHYPERT'
htn_3 = htn_3.rename(columns={'HMHYPERT':'hypertension'})
htn = pd.concat([htn_1,htn_2,htn_3])           # bind all the rows
htn = drop_char(htn,'RID').reset_index(drop=True)   #drop the RID which could not be converted to integer
htn = htn.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates
htn.head(3)


# In[21]:


htn.info()


# check if there are hypertension positive samples in each file, to make sure that we are using the right files:
# 
# all three files have positive and negative samples. So we are using the right files.

# In[22]:


htn_1[htn_1['hypertension']==1].head(2)   


# In[23]:


htn_2[htn_2['hypertension']==1].head(2)


# In[24]:


htn_3[htn_3['hypertension']==1].head(2)


# #### Question: should each patient have only one hypertension label?

# In[26]:


vis_dict = pd.read_csv("VISITS_DICT.csv",sep = ';')       #VISCODE DICTIONARY
vis_dict.head(11)


# #### RID=3 is marked as hypertension positive at the first time (VISCODE = sc), in the later visits, as negative. RID=4 is labeled as positive at first and second visits (VISCODE=sc,bl). It is very much unlikely that, after 6 months, his/her hypertension is cured.  Hypertension is considered as can be controlled with medication, but it can not be cured. So hypertension data is unique to RID, not change by ' Phase' 'VISCODE' or other features. 

# In[27]:


htn[htn['RID']==3]   


# In[28]:


htn_short = htn.groupby(['RID']).sum().reset_index()
htn_short.loc[(htn_short.hypertension >= 1), 'hypertension'] = 1
htn_short.loc[(htn_short.hypertension == 0), 'hypertension'] = 0
htn_short.to_csv('htn_short.csv')
htn_short.head(3)


# ### OSA
# Till now,  sleep apnea can not be cured completely

# In[29]:


ass_diag_merge


# In[30]:


osa_1 = new_col_with_key(ass_diag_merge,key_osa,'OSA')    # extract key words from assessment diagnosis files
osa_2 = new_col_with_key(med_hist_merge,key_osa,'OSA')   # extract key words from medical history files
osa_merge = pd.concat([osa_1,osa_2])           # bind all the rows
osa_merge = drop_char(osa_merge,'RID').reset_index(drop=True)   #drop the RID which could not be converted to integer
osa_merge = osa_merge.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates
osa_merge.head(3)


# In[31]:


osa_merge.info()


# In[32]:


osa_1[osa_1['OSA']==1].head(2)   # both osa_1 and osa_2 contain positive OSA -> the files we used are OSA-relevant


# In[33]:


osa_2[osa_2['OSA']==1].head(2)


# #### Question: should each patient have only one OSA label? 

# check some patients
# 
# i.e.RID=999: 17 records/visit, but only one positive record (v06,some middle time point of the whole study),since sleep spnea is not curable, especially not in that short time.
# 
# RID=205: 17 records/visit, but only one positive record (m06, again some middle time point of the whole study)
# 
# So, each patient should have only one OSA label.

# In[34]:


osa_merge[osa_merge['RID']==205]   


# In[35]:


osa_short = osa_merge.groupby(['RID']).sum().reset_index()
osa_short.loc[(osa_short['OSA'] >= 1), 'OSA'] = 1
osa_short.loc[(osa_short['OSA'] == 0), 'OSA'] = 0
osa_short.to_csv('osa_short.csv')
osa_short.head(3)


# ### Insomnia
# Insomnia can be fully cured. 75% of individuals with acute insomnia were able to make a full recovery after about 12 months.

# In[93]:


adsxlist.info()


# In[91]:


insomnia_3.info()


# In[94]:


#ass_diag_merge2 = pd.concat([blchange,modhach]).reset_index(drop=True) # ass_diag_merge without adsxlist
#insomnia_1 = new_col_with_key(ass_diag_merge2,'insomnia','insomnia')   # extract key words from assessment diagnosis 2 files, because the ADSXLIST also has the insomnia col
#insomnia_2 = new_col_with_key(med_hist_merge,'insomnia','insomnia')   # extract key words from medical history files
insomnia_merge = adsxlist[com_col]  
insomnia_merge['insomnia']= adsxlist['AXINSOMN'] - 1 #take the column 'AXINSOMN' and convert 1->0, 2->1
#insomnia_3['RID'] = insomnia_3['RID'].astype(str)
# insomnia_3 = insomnia_3#.drop_duplicates()
#insomnia_merge = pd.concat([insomnia_1,insomnia_2,insomnia_3])           # bind all the rows
insomnia_merge = drop_char(insomnia_merge,'RID').reset_index(drop=True)   #drop the RID which could not be converted to integer
insomnia_merge = insomnia_merge.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates
def rename_value(value):
    if value not in [0,1]:
        return np.nan
    else:
        return value
insomnia_merge['insomnia'] = insomnia_merge['insomnia'].apply(rename_value)   # replace the values that are not 1 or 0 to Nan
insomnia_merge.head(3)


# In[95]:


insomnia_merge.info()


# In[96]:


insomnia_merge.groupby('insomnia').count()


# In[98]:


insomnia_merge[insomnia_merge['RID']==109].sort_values(by=['VISCODE']) 


# #### check some samples.
# RID=108: all records are insomnia positive.
# RID=3: all records are insomnia positive (bl showed up two times, once = 0, once 1).Many samples have similiar situation.
# 
# So, the strategie is, for the records with same RID and VISCODE, we should use the sum of the records, if sum>= 1,insomnia should be set as 1.
# 
# for the insomnia data, we need both RID and VISCODE features to link to other data.

# In[105]:


insomnia_merge[insomnia_merge['RID']==109] # under each VISCODE there are more than one insomnia record even with different answers.


# Is it right? Or should keep both?

# In[48]:


insomnia_short = insomnia_merge.groupby(['RID','VISCODE','Phase']).sum().reset_index()  # groupby the rows with same 'RID and VISCODE', and sum
insomnia_short.loc[(insomnia_short['insomnia'] >= 1), 'insomnia'] = 1 
insomnia_short


# In[49]:


insomnia_short.groupby('insomnia').count()


# In[50]:


insomnia_short.to_csv('insomnia_short.csv')
insomnia_short.head(3)


# ### Anxiety
# There's no way to completely cure any anxiety disorder

# In[51]:


anxiety_1 = new_col_with_key(ass_diag_merge,'anxi','anxiety')    # extract key words from assessment diagnosis files
#anxiety_1.loc[(ass_diag_merge.AXDPMOOD == 2), 'depression'] = 1
anxiety_2 = new_col_with_key(med_hist_merge,'anxi','anxiety')   # extract key words from medical history files
anxiety_merge = pd.concat([anxiety_1,anxiety_2])           # bind all the rows
anxiety_merge = drop_char(anxiety_merge,'RID').reset_index(drop=True)   #drop the RID which could not be converted to integer
anxiety_merge = anxiety_merge.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates
anxiety_merge.head(3)


# In[52]:


anxiety_1[anxiety_1['anxiety']==1].head(2) # both files have positive depression samples--> it is right to take these files for anxiety labels


# In[53]:


anxiety_2[anxiety_2['anxiety']==1].head(2)


# #### check some samples.
# RID=2, with VISCODE we could see that this patient has discontinuous anxiety positive, which is impossible to be true. So the 0 is just the time, his/her anxiety disorder is not recorded. 
# RID=4, similiar to RID=2. 
# 
# Each patient should have only one label for anxiety disorder.

# In[54]:


anxiety_merge[anxiety_merge['RID']==3]   


# In[55]:


anxiety_short = anxiety_merge.groupby(['RID']).sum().reset_index()
anxiety_short.loc[(anxiety_short['anxiety'] >= 1), 'anxiety'] = 1
anxiety_short.loc[(anxiety_short['anxiety'] == 0), 'anxiety'] = 0
anxiety_short.to_csv('anxiety_short.csv')
anxiety_short.head(3)


# In[56]:


anxiety_merge.groupby('anxiety').count()


# ### depression
# There's no cure for depression, but you still have plenty of options for treatment, all of which can improve your symptoms and minimize their impact on your daily life.

# In[57]:


dxsum = pd.read_csv('DXSUM_PDXCONV_ADNIALL.csv',sep=';')
depression_1 = new_col_with_key(ass_diag_merge,'depr','depression')    # extract key words from assessment diagnosis files
depression_1.loc[(ass_diag_merge.AXDPMOOD == 2), 'depression'] = 1
depression_2 = new_col_with_key(med_hist_merge,'depr','depression')   # extract key words from medical history files
depression_3 = dxsum[com_col + ['DXAPROB','DXAPOSS']]                     # from column'HMHYPERT'
depression_3.loc[(depression_3['DXAPOSS'] == '1')|(depression_3['DXAPROB'] == '1'), 'depression'] = 1
depression_3 = depression_3[com_col + ['depression']]
#htn_3 = htn_3.rename(columns={'HMHYPERT':'hypertension'})

depression_merge = pd.concat([depression_1,depression_2,depression_3])           # bind all the rows
depression_merge = drop_char(depression_merge,'RID').reset_index(drop=True)   #drop the RID which could not be converted to integer
depression_merge = depression_merge.drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])             # drop the duplicates
depression_merge.head(3)


# In[58]:


depression_1[depression_1['depression']==1].head(2)  # both files have positive depression samples--> it is right to take these files for depression labels


# In[59]:


depression_2[depression_2['depression']==1].head(2)


# In[60]:


depression_3[depression_3['depression']==1].head(2)


# #### check some examples, the feature 'depression' is again similiar to 'depression' and 'OSA', that each patient should have only one label.

# In[61]:


depression_merge[depression_merge['RID']==2]


# In[62]:


depression_short = depression_merge.groupby(['RID']).sum().reset_index()
depression_short.loc[(depression_short['depression'] >= 1), 'depression'] = 1
depression_short.loc[(depression_short['depression'] == 0), 'depression'] = 0
depression_short = depression_short.dropna()
depression_short.to_csv('depression_short.csv')
depression_short


# In[63]:


depression_short.info()


# ### GDSCALE

# In[64]:


gdscale = pd.read_csv('GDSCALE.csv')
gdscale.head(2)


# In[65]:


col_to_drop = ['ID','SITEID','VISCODE2','USERDATE','USERDATE2', 'EXAMDATE','GDSOURCE','GDUNABL','GDUNABSP','update_stamp']
gdscale_short = gdscale.drop(col_to_drop,axis=1).drop_duplicates().reset_index(drop=True).sort_values(by= ['RID']) 
gdscale_short['GDCAT'] = gdscale_short['GDTOTAL'].apply(lambda x: 3 if x > 9 else 2 if x > 4 else 1 if x > 0 else -4)
gdscale_short = drop_char(gdscale_short,'RID').reset_index(drop=True)  #drop the row where RID cant be converted to integer
gdscale_supershort = gdscale_short[com_col + ['GDTOTAL','GDCAT']]
gdscale_short.to_csv('gdscale_short.csv')
gdscale_supershort.to_csv('gdscale_supershort.csv')
gdscale_short
gdscale_supershort


# ### NEUROBAT
# 
# Neuropsychological Battery[ADNI1,GO,2]: Neurobattery scores (i.e. LIMMTOTAL (immediate recall total score), AVTOT1-AVTOT5 (Rey Auditory Verbal Learning Test scores)) (NEUROBAT.csv)

# In[66]:


neurobat = pd.read_csv('NEUROBAT.csv',sep=';')
neurobat.head(2)


# In[67]:


col_to_drop = ['ID','SITEID','VISCODE2','USERDATE','USERDATE2', 'EXAMDATE','update_stamp']
neurobat_short = neurobat.drop(col_to_drop,axis=1).drop_duplicates().reset_index(drop=True).sort_values(by= ['RID']) 
neurobat_short = drop_char(neurobat_short,'RID').reset_index(drop=True)  #drop the row where RID cant be converted to integer
neurobat_supershort = neurobat_short[com_col + ['LIMMTOTAL','AVTOT1','AVTOT2','AVTOT3','AVTOT4','AVTOT5']]
neurobat_short.to_csv('neurobat_short.csv')
neurobat_supershort.to_csv('neurobat_supershort.csv')
neurobat_short
neurobat_supershort


# ### ADNIMERGE  
# #### there are many interesting features, i.e. the brain volume? 

# In[68]:


adnimerge = pd.read_csv('ADNIMERGE.csv',sep=',')
adnimerge.head(2)


# In[69]:


adnimerge_short = adnimerge[['RID', 'PTID','ORIGPROT', 'VISCODE','DX_bl','DX', 'ABETA','TAU','PTAU','ABETA_bl','TAU_bl','PTAU_bl']].drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])  
adnimerge_short = drop_char(adnimerge_short,'RID').reset_index(drop=True)  #drop the row where RID cant be converted to integer
adnimerge_short = adnimerge_short.drop_duplicates()
adnimerge_short = adnimerge_short.rename(columns = {'ORIGPROT':'Phase'})
adnimerge_short.to_csv('adnimerge_short.csv')
adnimerge_short.info()


# In[70]:


adnimerge_short.head(3)


# ### demographics data (short version): each RID should have only unique record

# In[71]:


demographic_short = adnimerge[['RID','PTGENDER','PTETHCAT','PTRACCAT','AGE']].drop_duplicates().reset_index(drop=True).sort_values(by= ['RID'])
demographic_short = drop_char(demographic_short,'RID').reset_index(drop=True)  #drop the row where RID cant be converted to integer
demographic_short.to_csv('demographic_short.csv')
demographic_short.head(3)


# In[72]:


demographic_short.info() == demographic_short.groupby(['RID']).count().info()  # True: each RID has only one record in this file.


# ### DXSUM_PDXCONV_ADNIALL: Diagnostic information

# In[73]:


dxsum.head(2)


# In[74]:


diag_short = dxsum[['Phase', 'RID', 'VISCODE','PTID','DXCHANGE', 'DXCURREN', 'DIAGNOSIS']].drop_duplicates().reset_index(drop=True).sort_values(by= ['RID']) 
diag_short = drop_char(diag_short,'RID').reset_index(drop=True)  #drop the row where RID cant be converted to integer
diag_short.to_csv('diag_adniall_short.csv')
diag_short.head(2)


# In[75]:


diag_short.info()


# Quick recap:
# - ADNI1: DXCURREN 1=NL; 2=MCI; 3=AD
# - ADNIGO/2: DXCHANGE    
#     1=Stable: NL to NL;    
#     2=Stable: MCI to MCI;  
#     3=Stable: Dementia to Dementia;   
#     4=Conversion: NL to MCI;   
#     5=Conversion: MCI to Dementia; 
#     6=Conversion: NL to Dementia;  
#     7=Reversion: MCI to NL;   
#     8=Reversion: Dementia to MCI;   
#     9=Reversion: Dementia to NL    
# - ADNI3: DIAGNOSIS 1=CN; 2=MCI; 3=Dementia

# #### ABETA 40; ABETA 42; from UPENNPLASMA.csv

# In[76]:


#up_pl = pd.read_csv('UPENNPLASMA.csv',sep = ';')
#up_pl


# ### join dataframes

# In[77]:


# by 'RID'
# hypertension, osa, anxiety, depression, demographics
htn_osa = pd.merge(htn_short,osa_short, how = 'outer', on = 'RID') 
htn_osa_anx = pd.merge(htn_osa,anxiety_short, how = 'outer', on = 'RID') 
htn_osa_anx_depr = pd.merge(htn_osa_anx,depression_short, how = 'outer', on = 'RID') 
htn_osa_anx_depr_demo = pd.merge(htn_osa_anx_depr,demographic_short, how = 'outer', on = 'RID') 

htn_osa_anx_depr_demo.to_csv('htn_osa_anx_depr_demograph_by_RID.csv')
htn_osa_anx_depr_demo


# In[106]:


# by 'Phase','RID', 'VISCODE'
# ADNIMERGE, DXSUM_PDXCONV_ADNIALL, NPI, INSOMNIA,GDSCALE, NEUROBAT  
adni_diag= pd.merge(adnimerge_short,diag_short, how = 'outer', on = ['Phase','RID','VISCODE','PTID']) 
adni_diag_npi = pd.merge(adni_diag,npi_short, how = 'outer', on = com_col) 
adni_diag_npi_insomnia = pd.merge(adni_diag_npi,insomnia_short, how = 'outer', on = com_col) 
adni_diag_npi_insomnia_GD = pd.merge(adni_diag_npi_insomnia,gdscale_short, how = 'outer', on = com_col) 
adni_diag_npi_insomnia_GD_neurobat = pd.merge(adni_diag_npi_insomnia_GD,neurobat_supershort, how = 'outer', on = com_col)
adni_diag_npi_insomnia_GD_neurobat.to_csv('adni_diag_npi_insomnia_GD_neurobat_by_phase_RID_VISCODE.csv')


# In[107]:


main_file = pd.merge(htn_osa_anx_depr_demo,adni_diag_npi_insomnia_GD_neurobat, how = 'outer', on='RID' )
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

# In[80]:


main_file.loc[(main_file['DIAGNOSIS'] == 1.0)|(main_file['DXCURREN']==1.0), 'DX'] = 'CN'
main_file.loc[(main_file['DIAGNOSIS'] == 2.0)|(main_file['DXCURREN']==2.0), 'DX'] = 'MCI'
main_file.loc[(main_file['DIAGNOSIS'] == 3.0)|(main_file['DXCURREN']==3.0), 'DX'] = 'AD'
main_file = main_file.drop(['DIAGNOSIS','DXCURREN'],axis=1)
main_file = main_file[main_file.VISCODE != 'f']   # drop the rows where VISCODE == 'f'
main_file


# In[81]:


dx_bl = main_file[main_file['VISCODE']=='bl'][['RID','DX']]   # diagnosis baseline
dx_bl.columns = ['RID','DX_bl']
dx_bl['DX_bl'].unique()


# In[82]:


main_file = main_file.drop(['DX_bl'],axis=1)   # each RID should have only one diagnosis baseline
main_file = main_file.merge(dx_bl,how='left',on='RID')
main_file['DXCHANGE'].unique()


# In[83]:


main_file['DX_bl'].unique()


# In[84]:


main_file['DX'].unique()


# In[85]:


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
main_file.to_csv('main_file.csv')
main_file['DXCHANGE'].unique()


# In[86]:


main_file.info()

