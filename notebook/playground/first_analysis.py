# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 19:43:58 2022

@author: Antonia
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('main_file.csv')


#%%

specs = ['Phase', 'RID', 'DIAGNOSIS','hypertension', 'depression', 'insomnia', 'anxiety', 'OSA', 'GDTOTAL','NPIKTOT', 'TAU', 'PTAU', 'ABETA']

df_small = df[specs]

df_small['GDCAT'] = df_small['GDTOTAL'].apply(lambda x: 3 if x > 9 else 2 if x > 4 else 1 if x > 0 else -4)
df_small['DIAGNOSIS'] = df_small['DIAGNOSIS'].map({1: "NC", 2: 'MCI', 3: 'AD'})

df_small = df_small.replace('-4', np.nan)


for col in columns: 
    df_small.ABETA =  df_small.ABETA.replace('<', np.nan, regex = True)
df_small.ABETA =  df_small.ABETA.replace('>', np.nan, regex = True)
df_small.ABETA = df_small.ABETA.astype(float)
df_small.TAU =  df_small.TAU.replace('<', '', regex = True)
df_small.TAU =  df_small.TAU.replace('>', '', regex = True)
df_small.TAU = df_small.TAU.astype(float)

#%%

df_change = df.loc[df.RID == 25]


print(df.DIAGNOSIS.value_counts())
print(df.DIAGNOSIS.isna().sum())

#%%


sns.scatterplot(x = df_small.ABETA, y = df_small.TAU )
plt.show()


sns.boxplot(y = df_small.ABETA)
plt.show()
#sns.boxplot(x = df_small.TAU)


sns.histplot(data=df_small.TAU)
sns.histplot(data=df_small.ABETA)
plt.show()

#ArithmeticErrorsns.histplot(data=df_small.DIAGNOSIS)
#%%


#%%


#print(df_small.corr())



# Cross tabulation between GENDER and APPROVE_LOAN
depr_gt=pd.crosstab(index=df_small['DIAGNOSIS'],columns=df_small['GDCAT'])
print(depr_gt)


insomnia_ct=pd.crosstab(index=df_small['DIAGNOSIS'],columns=df_small['insomnia'])
print(insomnia_ct)

OSA_ct=pd.crosstab(index=df_small['DIAGNOSIS'],columns=df_small['OSA'])
print(OSA_ct)



 


#%%

# sns.catplot(df_small)

# sns.heatmap(df_small.corr(), square=True, cmap='RdYlGn')
# plt.show()


y = df['DIAGNOSIS']
y.s()
pd.plotting.scatter_matrix(df_small[['ABETA','TAU', 'PTAU']], c = y )
plt.show()

#%%
df2 = df.loc[(df['insomnia'] == 1) &(df['OSA'] == 1) , ['Phase','VISCODE','RID','NPIKTOT', 'insomnia', 'OSA']]

df3 = df.loc[df.RID == 307]
df3 = df3.dropna(axis=1, how='all') 

