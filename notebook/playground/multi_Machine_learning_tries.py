# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:32:00 2022

@author: Antonia
"""
#%% install libraries, lopad data
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt


#actual ML part: importing libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
#%% loading data 
df = pd.read_csv('main_file_1206_new.csv')

npicat = CategoricalDtype(categories=['high', 'low'])

df['NPIKTOT'] = pd.to_numeric(df.NPIKTOT, errors='coerce')
df['NPIKCAT'] = df.NPIKTOT.apply(lambda x: 'high' if x >= 3 else 'low')
df['NPIKCAT'] = df['NPIKCAT'].astype(npicat)

df['Hippo_ratio'] =  df.Hippocampus / df.Hippocampus_bl
df['venti_ratio'] =  df.Ventricles /df.Ventricles_bl 
df['Entorhinal_ratio'] = df.Entorhinal / df.Entorhinal_bl
df['whole_ratio'] = df.WholeBrain / df.WholeBrain_bl 


#%% Statistic shit: av. Brainsize for age etc


adni1 = df.loc[(df.Phase.isin(['ADNI1', 'ADNIGO'])), ['Phase', 'RID','VISCODE2','EXAMDATE','PTDOBYY', 'DIAG_CHANGED', 
                       'PATIENT_DIAG_GROUP','PATIENT_FIRST_DIAG','PTETHCAT','DXCONFID', 'GDTOTAL', 'AXINSOMN', 'NPIKTOT','DIAG',
                       'Hippo_ratio', 'venti_ratio', 'Entorhinal_ratio', 'whole_ratio']]

adni1 = adni1.loc[((adni1.DIAG_CHANGED == True) & (adni1.DIAG == 'AD'))|((adni1.PATIENT_FIRST_DIAG == 'AD') & (adni1.VISCODE2 == 'sc'))]
adni1 = adni1.dropna(thresh = 12) #Remove rows with lots of missing values
adni1 = adni1.dropna(subset = 'DIAG')

adni1['EXAMDATE'] = adni1.EXAMDATE.astype(str)
adni1['EXAMDATE'] = adni1.EXAMDATE.str.slice(0, 4).astype('int') 
adni1.PTDOBYY = adni1.groupby('RID').PTDOBYY.ffill()
adni1.PTDOBYY = adni1.groupby('RID').PTDOBYY.bfill()
adni1 = adni1.dropna(subset = 'PTDOBYY')

adni1['age'] = adni1.EXAMDATE - adni1.PTDOBYY

#min_age = adni1.loc[((adni1.DIAG_CHANGED == True) & (adni1.DIAG == 'AD'))|((adni1.PATIENT_FIRST_DIAG == 'AD') & (adni1.VISCODE2 == 'sc'))]

df['exam'] = df[df.VISCODE2 == 'bl'].groupby('RID').lamda(EXAMDATE = df[df.VISCODE2 == 'bl'])

bl = df[['exam','VISCODE2']]
#Summary statistic grouped by AXINSOMN
print(adni1.groupby('AXINSOMN').age.describe())


sns.boxplot(y = adni1.age, x = adni1.AXINSOMN)
plt.show()
#%% linear regression for age, dependant on brainsize
ml =  df.loc[(df.Phase.isin(['ADNI1'])), ['Phase', 'RID','VISCODE2','EXAMDATE','PTDOBYY', 'DIAG_CHANGED', 
                       'PATIENT_DIAG_GROUP','PATIENT_FIRST_DIAG','PTETHCAT','DXCONFID', 'GDTOTAL', 'AXINSOMN', 'NPIKTOT','DIAG',
                       'Hippo_ratio', 'venti_ratio', 'Entorhinal_ratio', 'whole_ratio']]
ml = ml.dropna(thresh = 12) #Remove rows with lots of missing values
ml = ml.dropna(subset = 'DIAG')

ml['EXAMDATE'] = ml.EXAMDATE.astype(str)
ml['EXAMDATE'] = ml.EXAMDATE.str.slice(0, 4).astype('int') 
ml.PTDOBYY = ml.groupby('RID').PTDOBYY.ffill()
ml.PTDOBYY = ml.groupby('RID').PTDOBYY.bfill()
ml = ml.dropna(subset = 'PTDOBYY')
ml['age'] = ml.EXAMDATE - ml.PTDOBYY
ml_cols = ['age','Hippo_ratio', 'venti_ratio', 'Entorhinal_ratio', 'whole_ratio' ]

SS = StandardScaler()
ml_clean = ml[ml_cols].dropna()
#linreg_scaled = SS.fit_transform(linreg)


X = ml_clean[['Hippo_ratio', 'venti_ratio', 'Entorhinal_ratio', 'whole_ratio' ]]
y = ml_clean.age



def linreg(X,y):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42,stratify=(y))
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train) 
    y_pred = model.predict(X_test)
    model.score(X_train,y_train)
    # Compute R-squared
    r_squared = model.score(X_test, y_test)
    
    # Compute RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    
    
  
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    reg = LinearRegression()
    cv_results = cross_val_score(reg, X, y, cv=kf)

# #fitting the pca algorithm with our data
# pca=PCA().fit(linreg_scaled)
# #plotting the cumulative summation of the explained variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of components')
# plt.ylabel('Variance % for each components')
# plt.title('Explained variance')
# plt.show()



# pca=PCA(n_components=1)
# x_pca=pca.fit_transform(linreg_scaled)

linreg(X,y)



linreg(X1,y1)

sns.regplot(ml_clean.age, ml_clean.venti_ratio)
sns.regplot(ml_clean.age, ml_clean.Hippo_ratio)
plt.show()
#%%
from sklearn.neural_network import MLPRegressor
scaled = SS.fit_transform(ml_clean)


X1 = scaled[:,2:6]
y

X_train, X_test, y_train, y_test = train_test_split(X1,y, test_size = 0.2, random_state=42,stratify=(y))
mlp = MLPRegressor()
mlp.fit(X_train,y_train)
y_pred_train = mlp.predict(X_train) 
y_pred = mlp.predict(X_test)
print(mlp.score(X_train, y_train))



#%% linreg and plotting
from sklearn.model_selection import cross_val_predict

cv_insomn = ml.loc[ml.AXINSOMN == 1,ml_cols]
cv_no_insomn = ml.loc[ml.AXINSOMN == 0,ml_cols]


sns.regplot(cv_insomn.age, cv_insomn.Hippo_ratio, color = 'b')
sns.regplot(cv_no_insomn.age, cv_no_insomn.Hippo_ratio, color = 'g')
plt.show()
sns.regplot(cv_no_insomn.age, cv_no_insomn.venti_ratio)
sns.regplot(cv_insomn.age, cv_insomn.venti_ratio)
plt.show()
    
cv_insomn_clean = cv_insomn[ml_cols].dropna()
cv_no_insomn_clean = cv_no_insomn[ml_cols].dropna()

def cv(df):
    X = df[['Hippo_ratio', 'Entorhinal_ratio', 'whole_ratio' ]]
    y = df.age
    
    lr = LinearRegression()
    predicted = cross_val_predict(lr, X, y, cv=10)
    
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()
    print(cross_val_score(lr, X, y))

cv(cv_insomn_clean)
cv(cv_no_insomn_clean)


cv_AD = ml.loc[ml.PATIENT_DIAG_GROUP == 'AD-AD'][ml_cols].dropna()
cv_NL = ml.loc[(ml.AXINSOMN == 0)&(ml.PATIENT_DIAG_GROUP == 'NL-NL')][ml_cols].dropna()
cv(cv_AD)
cv(cv_NL)

sns.regplot(cv_AD.age, cv_AD.venti_ratio)
sns.regplot(cv_AD.age, cv_AD.Hippo_ratio)

#%% same for Adni2 (because different codes uuuggghhh)
adni2 = df.loc[(df.Phase =='ADNI2'), ['Phase', 'RID','VISCODE2','EXAMDATE','PTDOBYY', 'DIAG_CHANGED', 'USERDATE', 
                       'PATIENT_DIAG_GROUP','PATIENT_FIRST_DIAG','DXCONFID', 'GDTOTAL', 'AXINSOMN', 'NPIKTOT','DIAG',
                       'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
                       'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]

vs = adni2.loc[adni2.EXAMDATE.isna()]

#%%

adni1_ad = adni1[adni1.DIAG == 'AD']
adni1_grouped = adni1.groupby(adni1_ad.RID)

#%% Making Subset with relevant columns:

brainsizes = df.loc[:,['Phase','VISCODE2', 'RID', 'DIAG_CHANGED',
                      'PATIENT_DIAG_GROUP','DXCONFID', 'NPIKTOT','DIAG',
                       'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
                       'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']]


descript = brainsizes.describe()

#%% various crosstables

cross_tab_certainty = pd.crosstab(index=df['DIAG'],
                             columns=df['DXCONFID'],
                             normalize="index")

print(cross_tab_certainty)


ct_insomn = pd.crosstab(index=df['NPIKCAT'],
                             columns=df['AXINSOMN'],
                             normalize="index")

print(ct_insomn)

npis = df[df.NPIKTOT.notnull()]
npid = npis.describe()


#%% Make datasets for normal /mci / ad brains
nl_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'NL-NL']
ad_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'AD-AD']
mci_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'MCI-MCI']

brainsizes['Hippo_ratio'] =  brainsizes.Hippocampus / brainsizes.Hippocampus_bl
brainsizes['venti_ratio'] =  brainsizes.Ventricles /brainsizes.Ventricles_bl 
brainsizes['Entorhinal_ratio'] = brainsizes.Entorhinal / brainsizes.Entorhinal_bl
brainsizes['whole_ratio'] = brainsizes.WholeBrain / brainsizes.WholeBrain_bl 


cross_tab_certainty = pd.crosstab(index=df['DIAG'],
                             columns=df['DXCONFID'],
                             normalize="index")

print(cross_tab_certainty)
#%% Diagram Boxplot & Lineplot for brainvolume vs insomnia
cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']
brainsizes = brainsizes[brainsizes.Phase == 'ADNI1']
nl_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'NL-NL']
ad_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'AD-AD']
mci_brains = brainsizes[brainsizes.PATIENT_DIAG_GROUP == 'MCI-MCI']  

for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.violinplot(x = nl_brains.AXINSOMN, y = nl_brains[i], ax = ax1)
    sns.lineplot(nl_brains.VISCODE2, nl_brains[i], hue = nl_brains.AXINSOMN, ax = ax4).set(xlabel = 'NL')
    sns.violinplot(x = mci_brains.AXINSOMN, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(mci_brains.VISCODE2, mci_brains[i], hue = mci_brains.AXINSOMN, ax = ax5).set(yticklabels=[], xlabel = 'MCI')
    sns.violinplot(x = ad_brains.AXINSOMN, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(ad_brains.VISCODE2, ad_brains[i], hue = ad_brains.AXINSOMN, ax = ax6).set(yticklabels=[], xlabel = 'AD')
    plt.show()
    
#%% Diagram 
cols = ['Hippo_ratio','venti_ratio', 'Entorhinal_ratio', 'whole_ratio']
nl_brains = brainsizes[brainsizes.DIAG == 'NL']
ad_brains = brainsizes[brainsizes.DIAG == 'AD']
mci_brains = brainsizes[brainsizes.DIAG == 'MCI']  

for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.violinplot(x = nl_brains.AXINSOMN, y = nl_brains[i], ax = ax1)
    sns.lineplot(nl_brains.VISCODE2, nl_brains[i], hue = nl_brains.AXINSOMN, ax = ax4).set(xlabel = 'NL')
    sns.violinplot(x = mci_brains.AXINSOMN, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(mci_brains.VISCODE2, mci_brains[i], hue = mci_brains.AXINSOMN, ax = ax5).set(yticklabels=[], xlabel = 'MCI')
    sns.violinplot(x = ad_brains.AXINSOMN, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(ad_brains.VISCODE2, ad_brains[i], hue = ad_brains.AXINSOMN, ax = ax6).set(yticklabels=[], xlabel = 'AD')
    plt.show()


for i in cols:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20,10))
    sns.violinplot(x = nl_brains.AXINSOMN, y = nl_brains[i], ax = ax1)
    sns.lineplot(nl_brains.VISCODE2, nl_brains[i], hue = nl_brains.AXINSOMN, ax = ax4).set(xlabel = 'NL')
    sns.violinplot(x = mci_brains.AXINSOMN, y = mci_brains[i], ax = ax2).set(yticklabels=[])
    sns.lineplot(mci_brains.VISCODE2, mci_brains[i], hue = mci_brains.AXINSOMN, ax = ax5).set(yticklabels=[], xlabel = 'MCI')
    sns.violinplot(x = ad_brains.AXINSOMN, y = ad_brains[i], ax = ax3).set(yticklabels=[])
    sns.lineplot(ad_brains.VISCODE2, ad_brains[i], hue = ad_brains.AXINSOMN, ax = ax6).set(yticklabels=[], xlabel = 'AD')
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

#%% histogram brainsize for insomnia

def histplot_total(columns):
    nl_brains_insom = nl_brains.loc[nl_brains.AXINSOMN == 1]
    #ad_brains_insom = ad_brains.loc[nl_brains.AXINSOMN == 1]
    #mci_brains_insom = mci_brains.loc[nl_brains.AXINSOMN == 1]
    for c in columns:
        _ = plt.boxplot(nl_brains_insom[c])
        #_ = plt.hist(nl_brains[c],  bins = 20, color = 'g', alpha = .5, label = 'NL')
        #plt.legend()
        plt.title(c)
        plt.show()
        
histplot_total(cols)
nl_brains_insom = nl_brains.loc[nl_brains.AXINSOMN == 1]

#%% boxpot brainvolumes per diagnosis comparins insomnia and not insomnia
for c in cols: 
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    sns.boxplot(y = nl_brains[c], x = nl_brains.AXINSOMN, ax = (ax1))
    ax1.set_title('nl')
    sns.boxplot(y = mci_brains[c], x = mci_brains.AXINSOMN, ax = (ax2))
    ax2.set_title('mci')
    sns.boxplot(y = mci_brains[c], x = mci_brains.AXINSOMN, ax = (ax3))
    ax3.set_title('ad')
    plt.title('c')
    plt.show()

#%% boxplots brainvolumne 

g = sns.FacetGrid(brainsizes, col = 'PATIENT_DIAG_GROUP', row = 'AXINSOMN')
g.map(sns.boxplot,'WholeBrain')



g = sns.FacetGrid(brainsizes, col = 'DIAG', row = 'AXINSOMN')
g.map(sns.boxplot,'WholeBrain')



#%% Histogram using DIAG instead of Patient-Diag-Group

nl_brains = brainsizes[brainsizes.DIAG == 'NL']
ad_brains = brainsizes[brainsizes.DIAG == 'AD']
mci_brains = brainsizes[brainsizes.DIAG == 'MCI']

def histplot_total(columns):
    for c in columns:
        _ = plt.hist(nl_brains[c], bins = 20, color = 'g', alpha = .5, label = 'NL')
        _ = plt.hist(ad_brains[c], bins = 20, color = 'r', alpha = .5, label = 'AD')
        _ = plt.hist(mci_brains[c], bins = 20, color = 'b', alpha = .5, label = 'MCI')
        plt.legend()
        plt.title(c + 'Base Diagnosis')
        plt.show()


cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']
cols_bl = ['Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']


histplot_total(cols)

#%% random function for histplots?
def histplots_diag(columns, baselines):
    for c in columns:
        for b in baselines:
            fig,(ax1, ax2, ax3) = plt.subplots(3,1)
            ax1.hist(nl_brains[c], bins = 20, color = 'g', alpha = .5)
            ax1.hist(nl_brains[b], bins = 20, color = 'b', alpha = .5)
            ax1.set_title(c+b+ ' Normal')
            ax2.hist(ad_brains[c], bins = 20, color = 'g', alpha = .5)
            ax2.hist(ad_brains[b], bins = 20, color = 'b', alpha = .5)
            ax2.set_title(c +b+' Alzheimer')
            ax3.hist(mci_brains[c], bins = 20, color = 'g', alpha = .5)
            ax3.hist(mci_brains[b], bins = 20, color = 'b', alpha = .5)
            ax1.set_title(c+b+' MCI')
            plt.show()
            
        
        # _ = plt.hist(mci_brains[c], bins = 20)
        # _ = plt.hist(ad_brains[c], bins = 20, color = 'r')
        # plt.title(c)




#%% make dataset & filter the uncertain diagnoses out
rel_cols = ['Phase','VISCODE2', 'RID', 'DIAG_CHANGED', 'DIAG','PATIENT_DIAG_GROUP', 
            'DXCONFID', 'GDTOTAL', 'AXINSOMN', 'NPIKTOT', 
            'ABETA', 'PTAU', 'TAU', 'ABETA_bl', 'PTAU_bl', 'TAU_bl',
            'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
            'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']

ml_df = df[rel_cols]
ml_df1 = ml_df.loc[(ml_df.DXCONFID != 'uncertain') & (ml_df.Phase == 'ADNI1') & (ml_df.VISCODE2 != 'sc')]
ml_df2 = ml_df.loc[(ml_df.DXCONFID != 'uncertain') & (ml_df.Phase == 'ADNI2')]

nl_ml = ml_df[ml_df.PATIENT_DIAG_GROUP == 'NL-NL']
ad_ml = ml_df[ml_df.PATIENT_DIAG_GROUP == 'AD-AD']
mci_ml = ml_df[ml_df.PATIENT_DIAG_GROUP == 'MCI-MCI']

test = ml_df[ml_df.RID == 475]


#%% k-means clustering for Adni1 (failure)

ml_cols = [ 
'ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ]

SS = StandardScaler()
ml_df_clean = ml_df1[ml_cols].dropna()
ml_df_scaled = SS.fit_transform(ml_df_clean)



#fitting the pca algorithm with our data
pca=PCA().fit(ml_df_clean)
#plotting the cumulative summation of the explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Variance % for each components')
plt.title('Explained variance')
plt.show()



pca=PCA(n_components=7)
x_pca=pca.fit_transform(ml_df_clean)

kmeans = KMeans(n_clusters=2)
y_kmeans=kmeans.fit_predict(x_pca)#Plotting through K-Means
plt.scatter(x_pca[y_kmeans==0,0],x_pca[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x_pca[y_kmeans==1,0],x_pca[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of data')
plt.legend()
plt.show()


ml_df_clean['cluster'] = y_kmeans

#%% PCA / Kmeans for ADNI2 with NPIKTot

ml_cols = ['NPIKTOT', 
'ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']


SS = StandardScaler()
ml_df_scaled = ml_df2[ml_cols].dropna(axis = 1)
ml_df_scaled = SS.fit_transform(ml_df_scaled)

#fitting the pca algorithm with our data
pca=PCA().fit(ml_df_scaled)
#plotting the cumulative summation of the explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Variance % for each components')
plt.title('Explained variance')
plt.show()
pca=PCA(n_components=8)
pca.fit(ml_df_scaled)
x_pca=pca.transform(ml_df_scaled)

kmeans = KMeans(n_clusters=2)
y_kmeans=kmeans.fit_predict(x_pca)#Plotting through K-Meansplt.scatter(x_pca[y_kmeans==0,0],x_pca[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x_pca[y_kmeans==1,0],x_pca[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of data')
plt.legend()
plt.show()



#%% trying clustering for the diagnosis

ml_cols = ['DIAG', 
'ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ]

ml_df1_dummies = pd.get_dummies(ml_df1[ml_cols], drop_first=True)

SS = StandardScaler()
ml_df_scaled = ml_df1_dummies.dropna()
ml_df_scaled = SS.fit_transform(ml_df_scaled)


#fitting the pca algorithm with our data
pca=PCA().fit(ml_df_scaled)
#plotting the cumulative summation of the explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Variance % for each components')
plt.title('Explained variance')
plt.show()

pca=PCA(n_components=7)
x_pca=pca.fit_transform(ml_df_scaled)

kmeans = KMeans(n_clusters=3)
y_kmeans=kmeans.fit_predict(x_pca)
#Plotting through K-Means
plt.scatter(x_pca[y_kmeans==0,0],x_pca[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x_pca[y_kmeans==1,0],x_pca[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(x_pca[y_kmeans==2,0],x_pca[y_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of data')
plt.legend()
plt.show()


#%% trying to predict insomnia based on other stuff

ml_cols = ['AXINSOMN','ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']

ml_df1_clean = ml_df1[ml_cols]
ml_df1_clean = ml_df1_clean.dropna(thresh=9) 
ml_df1_clean = ml_df1_clean.dropna()


X = SS.fit_transform(ml_df1_clean[['ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']])
y = ml_df1_clean.AXINSOMN

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42,stratify=(y))
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

score = pd.DataFrame({'Train':[accuracy_score(y_train, y_pred_train),
                               recall_score(y_train, y_pred_train),
                               precision_score(y_train, y_pred_train)],
                      'Test':[accuracy_score(y_test, y_pred),
                              recall_score(y_test, y_pred),
                              precision_score(y_test, y_pred)]})




#%% linear regression for Ventricles

ml_cols = ['AXINSOMN','ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']

ml_df1_clean = ml_df1[ml_cols]
ml_df1_clean = ml_df1_clean.dropna(thresh=9) 
ml_df1_clean = ml_df1_clean.dropna()


X = SS.fit_transform(ml_df1_clean[['ABETA', 'PTAU', 'TAU', 
'AXINSOMN', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']])
y = ml_df1_clean.Ventricles

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state=42,stratify=y)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)


from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
