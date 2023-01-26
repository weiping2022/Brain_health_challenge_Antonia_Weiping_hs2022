import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt

df_complete = pd.read_csv('df_complete.csv')
#%%

df_complete['AGE'] = (2022 - df_complete.PTDOBYY)

sns.countplot(df_complete.PTGENDER)
plt.show()


sns.violinplot(df_complete.PTGENDER, df_complete.AGE)
plt.show()

sns.countplot(df_complete.DIAG, hue = df_complete.PTGENDER)
plt.show()


age_diag = df_complete.loc[df_complete.DIAG == 'AD']


age_diag.groupby('PTGENDER').mean()


#%% Visualisations


sns.boxplot(df_complete.DIAG, df_complete.GDTOTAL)
plt.show()

sns.boxplot(df_complete.GDCAT, df_complete.NPIKTOT)
plt.show()

sns.boxplot(x=df_complete.DIAG, y = df_complete.GDTOTAL)
plt.show()

sns.boxplot(x=df_complete.DIAG, y = df_complete.NPIKTOT)
plt.show()

sns.countplot(df_complete.DIAG_CHANGED)
plt.show()

sns.boxplot(df_complete.PATIENT_DIAG_GROUP, df_complete.AXDPMOOD)
plt.show()



fig, ax = plt.subplots(figsize=(80,80))
sns.heatmap(df_complete.corr(), square=True, cmap='YlGnBu', min = 0.5)
plt.show()





#%% finding the most depressed person


print(df_complete.loc[max(df_complete.GDTOTAL)])
non_depressed = df_complete.loc[(df_complete.GDTOTAL == 0)&
                                (df_complete.DIAG == 'AD')&
                                (df_complete.Phase == 'ADNI1')]
      
depressed_as_fuck = df_complete[(df_complete.RID == 926)&(df_complete.Phase == 'ADNI1')]

compare = pd.concat([depressed_as_fuck,non_depressed])

phases = list(compare.VISCODE2.unique())
patients = []

for phase in phases:
    for i, row in compare.iterrows():
        if row.VISCODE2 == phase:
            patients.append(row.RID)
        
newlist = [x for x in patients if patients.count(x) >= (len(phases)-2)]
s = set(newlist)

#♥compare = compare.loc[compare['RID'].isin(list(s))]



fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1)
sns.lineplot(x = compare.VISCODE2, y = compare.Hippocampus, hue = compare.RID, ax = ax1)
sns.lineplot(x = compare.VISCODE2, y = compare.WholeBrain,hue = compare.RID, ax = ax2)
sns.lineplot(x = compare.VISCODE2, y = compare.MidTemp,hue = compare.RID, ax = ax3)
sns.lineplot(x = compare.VISCODE2, y = compare.Ventricles, hue = compare.RID,ax = ax4)
plt.show()

#%%

brainsizes = df_complete.loc[df_complete.Phase == 'ADNI1', ['VISCODE2', 'RID', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'DIAG_CHANGED', 'PATIENT_DIAG_GROUP', 'GDTOTAL']]
brainsizes['GDCAT'] = brainsizes['GDTOTAL'].apply(lambda x: 3 if x > 9 else 2 if x > 4 else 1 if x > 0 else np.nan)
values = brainsizes[['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']]

brainsizes1 = df_complete.loc[df_complete.Phase == 'ADNI1', ['VISCODE2', 'RID', 'DIAG_CHANGED', 'PATIENT_DIAG_GROUP', 'GDTOTAL',
                                                             'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
                                                             'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]
viscode2_type = CategoricalDtype(categories=['sc', 'scmri', 'bl', 'mc03', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48',
                                 'm54', 'm60', 'm66', 'm72', 'm78', 'm84', 'm90', 'm96', 'm108', 'm120', 'm132', 'm144', 'm156', 'm168', 'm180', 'f'], ordered=True)


brainsizes1['VISCODE2'] = brainsizes1['VISCODE2'].astype(viscode2_type)


brainsizes1['GDCAT'] = brainsizes1['GDTOTAL'].apply(lambda x: 3 if x > 9 else 2 if x > 4 else 1 if x > 0 else np.nan).astype('category')
values_bl = brainsizes1[['Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]

values = brainsizes1[['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']]



for i in values_bl:
    sns.lineplot(brainsizes1.VISCODE2, brainsizes1[i], hue = brainsizes1.GDCAT)
    plt.show()
    

#%%

brainsizes2 = df_complete.loc[df_complete.Phase == 'ADNI2', ['VISCODE2', 'RID', 'DIAG_CHANGED', 'PATIENT_DIAG_GROUP', 'GDTOTAL',
                                                             'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
                                                             'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]
viscode2_type = CategoricalDtype(categories=['sc', 'scmri', 'bl', 'mc03', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48',
                                 'm54', 'm60', 'm66', 'm72', 'm78', 'm84', 'm90', 'm96', 'm108', 'm120', 'm132', 'm144', 'm156', 'm168', 'm180', 'f'], ordered=True)


brainsizes2['VISCODE2'] = brainsizes2['VISCODE2'].astype(viscode2_type)


brainsizes2['GDCAT'] = brainsizes2['GDTOTAL'].apply(lambda x: 3 if x > 9 else 2 if x > 4 else 1 if x > 0 else np.nan)
values = brainsizes2[['Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]


for i in values:
    sns.lineplot(brainsizes2.VISCODE2, brainsizes2[i], hue = brainsizes2.GDCAT)
    sns.lineplot(brainsizes2.VISCODE2, brainsizes2[i])
    plt.show()
    
#%%
brainsizes = df_complete.loc[:,['VISCODE2', 'RID', 'DIAG_CHANGED', 'PATIENT_DIAG_GROUP', 'GDTOTAL', 'AXINSOMN', 'NPIKTOT','DIAG',
                                                             'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
                                                             'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]
viscode2_type = CategoricalDtype(categories=['sc', 'scmri', 'bl', 'mc03', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48',
                                 'm54', 'm60', 'm66', 'm72', 'm78', 'm84', 'm90', 'm96', 'm108', 'm120', 'm132', 'm144', 'm156', 'm168', 'm180', 'f'], ordered=True)


brainsizes['VISCODE2'] = brainsizes['VISCODE2'].astype(viscode2_type)


brainsizes['GDCAT'] = brainsizes['GDTOTAL'].apply(lambda x: 3 if x > 9 else 2 if x > 4 else 1 if x > 0 else np.nan).astype('category')
values = brainsizes[['Hippocampus', 'Hippocampus_bl', 'WholeBrain_bl', 'WholeBrain', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]    
#%%
    
for i in values:
    sns.lineplot(brainsizes.VISCODE2, brainsizes[i], hue = brainsizes.GDCAT)
    plt.xticks(rotation=45)
    plt.show()
    
    
    

#%% lineplot of brainsize vs brainsize_bl
values = brainsizes[['Hippocampus', 'Hippocampus_bl', 'WholeBrain_bl', 'WholeBrain', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]    
for i in values:
    sns.lineplot(brainsizes.VISCODE2, brainsizes[i], hue = brainsizes.AXINSOMN)
    plt.xticks(rotation=45)
    plt.show()

#%%

for i in values:
    sns.lineplot(brainsizes.VISCODE2, brainsizes[i], hue = brainsizes.DIAG)
    plt.xticks(rotation=45)
    plt.show()



#%%

sns.scatterplot(brainsizes.NPIKTOT, brainsizes.WholeBrain)

#%% countplots depression

firstvisit = df_complete[(df_complete.VISCODE2 == 'bl') & (df_complete.Phase == 'ADNI1')]

firstvisit.RID.unique()

ads = firstvisit[firstvisit.PATIENT_FIRST_DIAG == 'AD']

non_ads = firstvisit[firstvisit.PATIENT_FIRST_DIAG != 'AD']

ads.AXDPMOOD = ads.AXDPMOOD.astype(int)
ads.DXDEP = ads.DXDEP.astype(int)
depr = sum(ads[(ads.AXDPMOOD == 1) | (ads.DXDEP == 1)])



sns.countplot(ads.AXDPMOOD)
plt.show()


sns.countplot(ads.DXDEP)
sns.countplot(non_ads.DXDEP)
plt.show()

sns.countplot(non_ads.DXDEP)
plt.show()

sns.countplot(non_ads.AXDPMOOD)
plt.show()

#%%

unique_patients = df_complete.loc[df_complete.RID.unique()]

sns.barplot(x = unique_patients.PATIENT_FIRST_DIAG, y= unique_patients.AXINSOMN)

plt.title(" Insomnia unique")
plt.show()


sns.countplot(non_ads.AXINSOMN)
plt.show()


#%% Checking if AXINSOMN is available in all phases

cross_tab_prop = pd.crosstab(index=df_complete['AXINSOMN'],
                             columns=df_complete['DIAG'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))
#%% Heatmap with reduced dataset

cols_new = [ 'DIAG_CHANGED', 'PATIENT_DIAG_GROUP', 'GDTOTAL', 
          'ABETA', 'ABETA_bl', 'TAU', 'TAU_bl', 'PTAU', 'PTAU_bl', 'NPIKTOT', 'NPIKSEV',
          'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
          'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']

df_new = df_complete[cols_new]

sns.heatmap(df_new.corr(), square = True)


sns.heatmap(brainsizes.corr())
#%% OSA vs. diagnosis? (distribution)
OSA = pd.read_csv('osa_short.csv')
key_attrs = ['Phase', 'RID', 'VISCODE2']


additional_data = OSA.drop(columns=['ID', 'VISCODE', 'SITEID', 'USERDATE', 'USERDATE2', 'EXAMDATE', 'update_stamp'], errors='ignore')

df_complete = pd.merge(df_complete, additional_data,
                             how='outer', on='RID')


cross_tab_prop = pd.crosstab(index=df_complete['OSA'],
                             columns=df_complete['DIAG'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))


cross_tab2 = pd.crosstab(index=df_complete['DIAG'],
                             columns=df_complete['OSA'],
                             normalize="index")

cross_tab2.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))
#%% Chi Square result
from scipy.stats import chi2_contingency
print(chi2_contingency(cross_tab2)[0:3])
#%%

brainsizes.hist(figsize=(15,15),bins=20)

import plotly.express as px
df_cor = df_complete.corr()
px.imshow(df_cor, title = 'Correlations between testscales', zmin=0, zmax=1)




#%% OSA vs. brainsize

brainsizes = df_complete.loc[:,['VISCODE2', 'RID', 'DIAG_CHANGED', 'PATIENT_DIAG_GROUP', 'GDTOTAL', 'AXINSOMN', 'NPIKTOT','DIAG','OSA',
                                                             'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
                                                             'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]
viscode2_type = CategoricalDtype(categories=['sc', 'scmri', 'bl', 'mc03', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48',
                                 'm54', 'm60', 'm66', 'm72', 'm78', 'm84', 'm90', 'm96', 'm108', 'm120', 'm132', 'm144', 'm156', 'm168', 'm180', 'f'], ordered=True)


brainsizes['VISCODE2'] = brainsizes['VISCODE2'].astype(viscode2_type)

values = brainsizes[['Hippocampus', 'Hippocampus_bl', 'WholeBrain_bl', 'WholeBrain', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]    

for i in values:
    sns.lineplot(brainsizes.VISCODE2, brainsizes[i], hue = brainsizes.OSA)
    plt.xticks(rotation=45)
    plt.show()


sns.scatterplot(brainsizes.VISCODE2, brainsizes.Hippocampus_bl, hue = brainsizes.OSA)
plt.xticks(rotation=45)
plt.show()

sns.boxplot(brainsizes.VISCODE2, brainsizes.Hippocampus_bl, hue = brainsizes.AXINSOMN)
plt.xticks(rotation=45)



plt.show()


#%%
cross_tab2 = pd.crosstab(index=df_complete['PATIENT_DIAG_GROUP'],
                             columns=df_complete[''],
                             normalize="index")

cross_tab2.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))

#%% Proportin of diagnosis per race
df = df_complete
cross_tab_prop = pd.crosstab(index=df['PTRACCAT'],
                             columns=df['DIAG'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))
#%%
cross_tab_prop = pd.crosstab(index=df['PTRACCAT'],
                             columns=df['AXINSOMN'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))

cross_tab_prop = pd.crosstab(index=df['PTETHCAT'],
                             columns=df['AXINSOMN'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))



#%%proportion of diagnosis per ethnicity
cross_tab_prop = pd.crosstab(index=df['PTETHCAT'],
                             columns=df['DIAG'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', 
                    stacked=True, 
                    colormap='tab10', 
                    figsize=(10, 6))


#%%
df_cor = df.corr()

df_cor = df_cor[df_cor.abs() > .5].replace(1, np.nan)

#%%
fig, ax = plt.subplots(figsize=(80,80))
sns.heatmap(df_cor, cmap='RdYlGn',  square = False)
plt.show()

#%%


