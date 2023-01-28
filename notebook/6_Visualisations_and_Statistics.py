# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:46:52 2023

@author: Antonia
"""

#%% install libraries, lopad data
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
import random

#%%
df = pd.read_csv('main_file_1206_new.csv')

npicat = CategoricalDtype(categories=['high', 'low'])
df['NPIKTOT'] = pd.to_numeric(df.NPIKTOT, errors='coerce')
df['NPIKCAT'] = df.NPIKTOT.apply(lambda x: '1' if x >3 else '0')



#%% Statistic shit: av. Brainsize for AGE etc

#Making dataframe of age of people who were diagnosed with AD

df['sleep_deprived'] = 0
df.loc[(df['insomnia'] == 1) & (df['OSA'] == 1) & (df.NPIKTOT > 3), 'sleep_deprived'] = 1 
min_AGE = df.loc[((df.DIAG_CHANGED == True) & (df.DIAG == 'AD'))|((df.PATIENT_FIRST_DIAG == 'AD') & (df.VISCODE2 == 'sc'))]


#Hypothesis testing: are people with insomnia likely to develop AD at an earlier age? 
min_age_sample = min_AGE.loc[min_AGE.insomnia == 0].AGE.sample(n = 50, random_state = 420)
print(min_age_sample.describe())

sns.scatterplot(x = min_AGE.AGE, )
#Summary statistic grouped by insomnia
print(min_AGE.groupby('insomnia').AGE.describe())
print(min_AGE.groupby('OSA').AGE.describe())
print(min_AGE.groupby('NPIKCAT').AGE.describe())


print(df.groupby('insomnia').AGE.describe())


cross_tab_certainty = pd.crosstab(index=df['DIAG'],
                              columns=df['DXCONFID'],
                              normalize="index")

# print(cross_tab_certainty)


ct_insomn = pd.crosstab(index=df['NPIKCAT'],
                             columns=df['insomnia'],
                             normalize="index")

print(ct_insomn)


cross_tab_certainty = pd.crosstab(index=df['DIAG'],
                             columns=df['DXCONFID'],
                             normalize="index")

print(pd.crosstab(index=df['OSA'],
                              columns=df['DIAG'],
                              normalize="index"))


print(pd.crosstab(index=df['insomnia'],
                              columns=df['DIAG'],
                              normalize="index"))


print(pd.crosstab(index=df['NPIKCAT'],
                              columns=df['DIAG'],
                              normalize="index"))

npis = df[df.NPIKTOT.notnull()]
npid = npis.describe()

xbar = min_AGE.groupby('insomnia')['AGE']

df.groupby('insomnia').count()

import scipy.stats as stats 

def hypothesistest_age(criteria):
    group1 = min_AGE[min_AGE[criteria] == 0]['AGE'].dropna()
    group2 = min_AGE[min_AGE[criteria] == 1]['AGE'].dropna()
    return stats.ttest_ind(a=group1, b=group2, nan_policy = 'omit',alternative = 'greater')

print(hypothesistest_age('insomnia'))
print(hypothesistest_age('OSA'))
print(hypothesistest_age('NPIKCAT'))



statistics = df.groupby('OSA').describe()



fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (10,15))
sns.boxplot(y = min_AGE.AGE, x = min_AGE.insomnia, ax = ax1)
sns.boxplot(y = min_AGE.AGE, x = min_AGE.OSA, ax = ax2)
sns.boxplot(y = min_AGE.AGE, x = min_AGE.NPIKCAT, ax = ax3)
plt.show()

#%% Lineplots
cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']
nl_brains = df[df.PATIENT_DIAG_GROUP == 'NL-NL']
ad_brains = df[df.PATIENT_DIAG_GROUP == 'AD-AD']
mci_brains = df[df.PATIENT_DIAG_GROUP == 'MCI-MCI']

for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.violinplot(x = nl_brains.OSA, y = nl_brains[i], ax = ax1)
    sns.lineplot(x = nl_brains.AGE, y = nl_brains[i],hue = nl_brains.OSA, ax = ax4).set(xlabel = 'NL-NL')
    sns.violinplot(x = mci_brains.OSA, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(x = mci_brains.AGE, y = mci_brains[i], hue = mci_brains.OSA,alpha = 0.4, ax = ax5).set(yticklabels=[], xlabel = 'MCI-MCI')
    sns.violinplot(x = ad_brains.OSA, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(x = ad_brains.AGE, y = ad_brains[i], hue = ad_brains.OSA,alpha = 0.4, ax = ax6).set(yticklabels=[], xlabel = 'AD-AD')
    fig.suptitle('Comparison of brain volumne, separated by OSA')
    plt.show()

#%%
brainsizes = df[['Phase','VISCODE2', 'RID', 'DIAG_CHANGED',
                      'PATIENT_DIAG_GROUP', 'insomnia', 'NPIKTOT','DIAG',
                       'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
                       'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl',
                       'ratio_Hippocampus_bl','ratio_Ventricles_bl', 'ratio_Entorhinal_bl', 'ratio_WholeBrain_bl']]


descript = brainsizes.describe()

#%% Make  Boxplot & Lineplot for nl-nl / mci-mci / ad-ad patients for all phases
cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']
nl_brains = brainsizes.loc[(brainsizes.PATIENT_DIAG_GROUP == 'NL-NL') & brainsizes.VISCODE2.isin(['sc', 'bl', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 'm48',
                                                  'm54', 'uns1', 'f', 'nv', 'scmri', 'm03', 'm60', 'm66', 'm72'])]
ad_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'AD-AD']
mci_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'MCI-MCI']

for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.violinplot(x = nl_brains.insomnia, y = nl_brains[i], ax = ax1)
    sns.lineplot(x =nl_brains.VISCODE2, y =nl_brains[i], hue = nl_brains.insomnia, ax = ax4).set(xlabel = 'NL-NL')
    sns.violinplot(x = mci_brains.insomnia, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(x = mci_brains.VISCODE2, y = mci_brains[i], hue = mci_brains.insomnia, ax = ax5).set(yticklabels=[], xlabel = 'MCI-MCI')
    sns.violinplot(x = ad_brains.insomnia, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(x = ad_brains.VISCODE2, y =ad_brains[i], hue = ad_brains.insomnia, ax = ax6).set(yticklabels=[], xlabel = 'AD-AD')
    fig.suptitle('Comparison of brain volumne for different diagnosis groups')
    plt.show()

#%%  for brainvolume vs insomnia
cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']
brainsizes2 = brainsizes[brainsizes.Phase == 'ADNI2']
nl_brains = brainsizes2[brainsizes2.PATIENT_DIAG_GROUP == 'NL-NL']
ad_brains = brainsizes2[brainsizes2.PATIENT_DIAG_GROUP == 'AD-AD']
mci_brains = brainsizes2[brainsizes2.PATIENT_DIAG_GROUP == 'MCI-MCI']  

for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.violinplot(x = nl_brains.insomnia, y = nl_brains[i], ax = ax1)
    sns.lineplot(x = nl_brains.VISCODE2, y = nl_brains[i], hue = nl_brains.insomnia, ax = ax4).set(xlabel = 'NL')
    sns.violinplot(x = mci_brains.insomnia, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(x = mci_brains.VISCODE2, y = mci_brains[i], hue = mci_brains.insomnia, ax = ax5).set(yticklabels=[], xlabel = 'MCI')
    sns.violinplot(x = ad_brains.insomnia, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(x = ad_brains.VISCODE2,y =  ad_brains[i], hue = ad_brains.insomnia, ax = ax6).set(yticklabels=[], xlabel = 'AD')
    plt.show()
    
#%% Diagram 
cols = ['ratio_Hippocampus_bl','ratio_Ventricles_bl', 'ratio_Entorhinal_bl', 'ratio_WholeBrain_bl']
nl_brains = brainsizes[brainsizes.DIAG == 'NL']
ad_brains = brainsizes[brainsizes.DIAG == 'AD']
mci_brains = brainsizes[brainsizes.DIAG == 'MCI']  

for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.boxplot(x = nl_brains.insomnia, y = nl_brains[i], ax = ax1)
    sns.lineplot(x = nl_brains.VISCODE2, y = nl_brains[i], hue = nl_brains.insomnia, ax = ax4).set(xlabel = 'NL')
    sns.boxplot(x = mci_brains.insomnia, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(x = mci_brains.VISCODE2, y = mci_brains[i], hue = mci_brains.insomnia, ax = ax5).set(yticklabels=[], xlabel = 'MCI')
    sns.boxplot(x = ad_brains.insomnia, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(x = ad_brains.VISCODE2, y = ad_brains[i], hue = ad_brains.insomnia, ax = ax6).set(yticklabels=[], xlabel = 'AD')
    plt.show()


for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.violinplot(x = nl_brains.insomnia, y = nl_brains[i], ax = ax1)
    sns.lineplot(nl_brains.VISCODE2, nl_brains[i], hue = nl_brains.insomnia, ax = ax4).set(xlabel = 'NL')
    sns.violinplot(x = mci_brains.insomnia, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(mci_brains.VISCODE2, mci_brains[i], hue = mci_brains.insomnia, ax = ax5).set(yticklabels=[], xlabel = 'MCI')
    sns.violinplot(x = ad_brains.insomnia, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(ad_brains.VISCODE2, ad_brains[i], hue = ad_brains.insomnia, ax = ax6).set(yticklabels=[], xlabel = 'AD')
    plt.show()

sub = brainsizes.loc[(brainsizes.PATIENT_DIAG_GROUP.isin([ 'NL-NL', 'AD-AD', 'MCI-MCI'])) & brainsizes.VISCODE2.isin(['sc', 'bl', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42', 
                                                                                                                      'm48','m54'])]
                                                  

fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize = (20,10))
sns.lineplot(x = sub.VISCODE2, y = sub['WholeBrain'], hue = sub.PATIENT_DIAG_GROUP, ax = ax1).set(xlabel = 'WholeBrain')
sns.lineplot(x = sub.VISCODE2, y = sub['Hippocampus'], hue = sub.PATIENT_DIAG_GROUP, ax = ax2).set(xlabel = 'Hippocampus')
sns.lineplot(x = sub.VISCODE2, y = sub['Ventricles'], hue = sub.PATIENT_DIAG_GROUP, ax = ax3).set(xlabel = 'Ventricles')
plt.show()



fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize = (20,10))
sns.lineplot(x = sub.VISCODE2, y = sub['WholeBrain'], hue = sub.DIAG, ax = ax1).set(xlabel = 'WholeBrain')
sns.lineplot(x = sub.VISCODE2, y = sub['Hippocampus'], hue = sub.DIAG, ax = ax2).set(xlabel = 'Hippocampus')
sns.lineplot(x = sub.VISCODE2, y = sub['Ventricles'], hue = sub.DIAG, ax = ax3).set(xlabel = 'Ventricles')
plt.show()



fig, ((ax1, ax2)) = plt.subplots(1,2, figsize = (20,10))
sns.lineplot(x = sub.VISCODE2, y = sub['ratio_Hippocampus_bl'], hue = sub.DIAG, ax = ax1).set(xlabel = 'Hippocampus')
sns.lineplot(x = sub.VISCODE2, y = sub['ratio_Ventricles_bl'], hue = sub.DIAG, ax = ax2).set(xlabel = 'Ventricles')
plt.show()

#%% Histogram using Patient Diag Group
def histplot_total(columns):
    for c in columns:
        _ = plt.hist(nl_brains[c], bins = 20, color = 'g', alpha = .5, label = 'NL')
        _ = plt.hist(ad_brains[c], bins = 20, color = 'r', alpha = .5, label = 'AD')
        _ = plt.hist(mci_brains[c], bins = 20, color = 'b', alpha = .5, label = 'MCI')
        plt.legend()
        plt.title(c)
        plt.show()

cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']
cols_bl = ['Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']


histplot_total(cols)


#%% boxpot brainvolumes per diagnosis comparins insomnia and not insomnia
for c in cols: 
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    sns.boxplot(y = nl_brains[c], x = nl_brains.insomnia, ax = (ax1))
    ax1.set_title('nl')
    sns.boxplot(y = mci_brains[c], x = mci_brains.insomnia, ax = (ax2))
    ax2.set_title('mci')
    sns.boxplot(y = mci_brains[c], x = mci_brains.insomnia, ax = (ax3))
    ax3.set_title('ad')
    plt.title('c')
    plt.show()

#%% boxplots brainvolumne 

g = sns.FacetGrid(brainsizes, col = 'PATIENT_DIAG_GROUP', row = 'insomnia')
g.map(sns.boxplot,'WholeBrain')



g = sns.FacetGrid(brainsizes, col = 'DIAG', row = 'insomnia')
g.map(sns.boxplot,'WholeBrain')

