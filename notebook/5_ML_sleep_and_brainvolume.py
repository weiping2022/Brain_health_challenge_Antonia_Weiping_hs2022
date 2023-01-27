# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:48:31 2022

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
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier

#%% loading data 
df = pd.read_csv('main_file_1206_new.csv')

npicat = CategoricalDtype(categories=['high', 'low'])
df['NPIKTOT'] = pd.to_numeric(df.NPIKTOT, errors='coerce')
df['NPIKCAT'] = df.NPIKTOT.apply(lambda x: '1' if x >3 else '0')



#%% Statistic shit: av. Brainsize for AGE etc
import random
#Making dataframe of age of people who were diagnosed with AD

df['sleep_deprived'] = 0
df.loc[(df['insomnia'] == 1) & (df['OSA'] == 1) & (df.NPIKTOT > 3), 'sleep_deprived'] = 1 
min_AGE = df.loc[((df.DIAG_CHANGED == True) & (df.DIAG == 'AD'))|((df.PATIENT_FIRST_DIAG == 'AD') & (df.VISCODE2 == 'sc'))]
#%%
ml =  df.loc[(df.Phase.isin(['ADNI1'])), ['Phase', 'RID','VISCODE2','EXAMDATE','DIAG_CHANGED', 'AGE',
                       'PATIENT_DIAG_GROUP','PATIENT_FIRST_DIAG', 'insomnia', 'NPIKTOT','DIAG',
                       'ratio_Hippocampus_bl', 'ratio_Ventricles_bl', 'ratio_Entorhinal_bl', 'ratio_WholeBrain_bl']]
ml = ml.dropna(thresh = 12) #Remove rows with lots of missing values
ml = ml.dropna(subset = 'DIAG')


ml_cols = ['AGE','ratio_Hippocampus_bl', 'ratio_Ventricles_bl', 'ratio_Entorhinal_bl', 'ratio_WholeBrain_bl' ]

SS = StandardScaler()
ml_clean = ml[ml_cols].dropna()
#linreg_scaled = SS.fit_transform(linreg)


X = ml_clean[['ratio_Hippocampus_bl', 'ratio_Ventricles_bl', 'ratio_Entorhinal_bl', 'ratio_WholeBrain_bl' ]]
y = ml_clean.AGE



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

linreg(X,y)

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
#%%


cv_insomn = ml.loc[ml.insomnia == 1,ml_cols]
cv_no_insomn = ml.loc[ml.insomnia == 0,ml_cols]

   
cv_insomn_clean = cv_insomn[ml_cols].dropna()
cv_no_insomn_clean = cv_no_insomn[ml_cols].dropna()

def cv(df):
    X = df[['ratio_Hippocampus_bl', 'ratio_Entorhinal_bl', 'ratio_WholeBrain_bl' ]]
    y = df.AGE
    
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
cv_NL = ml.loc[(ml.insomnia == 0)&(ml.PATIENT_DIAG_GROUP == 'NL-NL')][ml_cols].dropna()
cv(cv_AD)
cv(cv_NL)

#%% make dataset & filter the uncertain diagnoses out
rel_cols = ['Phase','VISCODE2', 'RID', 'DIAG_CHANGED', 'DIAG','PATIENT_DIAG_GROUP', 
             'insomnia', 'NPIKTOT', 
            'ABETA', 'PTAU', 'TAU', 'ABETA_bl', 'PTAU_bl', 'TAU_bl',
            'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp' ,
            'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl']

ml_df = df[rel_cols]
#ml_df = ml_df.loc[(ml_df.DXCONFID != 'uncertain') & (ml_df.Phase == 'ADNI1') & (ml_df.VISCODE2 != 'sc')]
#ml_df2 = ml_df.loc[(ml_df.DXCONFID != 'uncertain') & (ml_df.Phase == 'ADNI2')]

nl_ml = ml_df[ml_df.PATIENT_DIAG_GROUP == 'NL-NL']
ad_ml = ml_df[ml_df.PATIENT_DIAG_GROUP == 'AD-AD']
mci_ml = ml_df[ml_df.PATIENT_DIAG_GROUP == 'MCI-MCI']

#%% k-means clustering for Adni1 (failure)

ml_cols = [ 
'ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']

SS = StandardScaler()
ml_df_clean = ml_df[ml_cols+['insomnia']].dropna()
ml_df_scaled = SS.fit_transform(ml_df_clean.iloc[:,:10])



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

print(pd.crosstab(index = ml_df_clean.insomnia, columns = ml_df_clean.cluster))



#%% trying clustering for the diagnosis

ml_df_dummies = pd.get_dummies(ml_df[ml_cols], drop_first=True)

SS = StandardScaler()
ml_df_clean = ml_df.dropna()
ml_df_dummies = pd.get_dummies(ml_df_clean[ml_cols], drop_first=True)
ml_df_scaled = SS.fit_transform(ml_df_dummies)


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


print(pd.crosstab(index = ml_df_clean.DIAG, columns = y_kmeans))




#%% Logistic Regression Model to predict insomnia
ml_cols = ['insomnia','ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']

ml_df_clean = df[ml_cols]
ml_df_clean = ml_df_clean.dropna(thresh=9) 
ml_df_clean = ml_df_clean.dropna()


X = SS.fit_transform(ml_df_clean[['ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']])
y = ml_df_clean.insomnia

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

print(pd.crosstab(index = y_train,
                  columns = y_pred_train))


#%% linear regression for Ventricles size based on other brain measurements and insomnia

ml_cols = ['insomnia','ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']

ml_df_clean = df[ml_cols]
ml_df_clean = ml_df_clean.dropna(thresh=9) 
ml_df_clean = ml_df_clean.dropna()


X = SS.fit_transform(ml_df_clean[['ABETA', 'PTAU', 'TAU', 
'insomnia', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']])
y = ml_df_clean.Ventricles

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state=42,stratify=y)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)




#%% logistic regression to predict diagnosis

ml_cols = ['DIAG','insomnia', 'OSA','ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']

ml_df_clean = df.loc[(df.PATIENT_DIAG_GROUP == 'MCI-AD')|(df.PATIENT_DIAG_GROUP == 'NL-AD'), ml_cols]
ml_df_clean = ml_df_clean.dropna(thresh=9) 
ml_df_clean = ml_df_clean.dropna()
ml_df_clean = pd.get_dummies(ml_df_clean)



X = SS.fit_transform(ml_df_clean[['ABETA', 'PTAU', 'TAU', 
'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']])
y = ml_df_clean.DIAG_AD

cv_scores = cross_val_score(model, X,y, cv = 10)
print(np.mean(cv_scores))

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

print(pd.crosstab(index = y_train,
                  columns = y_pred_train))

#%% logistic regression to predict diagnosis but using ratio of brain volumes

ml_cols = ['DIAG','insomnia', 'OSA','ABETA', 'PTAU', 'TAU', 
 'ratio_Ventricles_bl',
'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
'ratio_Fusiform_bl', 'ratio_ICV_bl', 'ratio_ABETA_bl', 'ratio_TAU_bl',
'ratio_PTAU_bl']

ml_df_clean = df[ml_cols]
ml_df_clean = ml_df_clean.dropna(thresh=9) 
ml_df_clean = ml_df_clean.dropna()
ml_df_clean = pd.get_dummies(ml_df_clean)


X = SS.fit_transform(ml_df_clean[['ABETA', 'PTAU', 'TAU', 'ratio_Ventricles_bl',
'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
'ratio_Fusiform_bl', 'ratio_ICV_bl', 'ratio_ABETA_bl', 'ratio_TAU_bl',
'ratio_PTAU_bl']])
y = ml_df_clean.DIAG_AD

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

print(pd.crosstab(index = y_train,
                  columns = y_pred_train))


print(score)

#%% logistic regression for DIAG Changed 
ml_cols = ['DIAG','insomnia', 'OSA', 
 'ratio_Ventricles_bl',
'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
'ratio_Fusiform_bl', 'ratio_ICV_bl','NPIKTOT']

ml_df_clean = df.loc[(df.PATIENT_DIAG_GROUP == 'MCI-AD')|(df.PATIENT_DIAG_GROUP == 'NL-AD'), ml_cols]
ml_df_clean = ml_df_clean.dropna(thresh=9) 
ml_df_clean = ml_df_clean.dropna()
ml_df_clean = pd.get_dummies(ml_df_clean)

X = SS.fit_transform(ml_df_clean[['ratio_Ventricles_bl',
'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
'ratio_Fusiform_bl', 'ratio_ICV_bl', 'NPIKTOT']])
y = ml_df_clean.DIAG_AD

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=420,stratify=(y))
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)


cv_scores = cross_val_score(model, X,y, cv = 10)
#%% logistic regression with undersampling
ml_df_clean = pd.read_csv('ml_df_clean_undersampled.csv')


def ml(params):
    X = SS.fit_transform(ml_df_clean[['ratio_Ventricles_bl',
    'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
    'ratio_Fusiform_bl', 'ratio_ICV_bl']+params])
    y = ml_df_clean.DIAG_AD

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=420,stratify=(y))
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)


    cv_scores = cross_val_score(model, X,y, cv = 10)
    return round(np.mean(cv_scores),6)

print(ml(['insomnia']))
print(ml(['OSA']))
print(ml(['NPIKTOT']))
print(ml(['NPIKTOT','insomnia', 'OSA']))
print(ml([]))

score = pd.DataFrame({'Train':[accuracy_score(y_train, y_pred_train),
                               recall_score(y_train, y_pred_train),
                               precision_score(y_train, y_pred_train)],
                      'Test':[accuracy_score(y_test, y_pred),
                              recall_score(y_test, y_pred),
                              precision_score(y_test, y_pred)]})

print(pd.crosstab(index = y_train,
                  columns = y_pred_train))


print(score)


#%% logistic regression for insomnia

X = SS.fit_transform(ml_df_clean[['ratio_Ventricles_bl',
'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
'ratio_Fusiform_bl', 'ratio_ICV_bl','insomnia','OSA','NPIKTOT']])
y = ml_df_clean.DIAG_AD

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

print(pd.crosstab(index = y_train,
                  columns = y_pred_train))


print(score)


#%% logistic regression for insomnia
ml_cols = ['sleep_deprived','ABETA', 'PTAU', 'TAU', 
 'ratio_Ventricles_bl',
'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
'ratio_Fusiform_bl', 'ratio_ICV_bl', 'ratio_ABETA_bl', 'ratio_TAU_bl',
'ratio_PTAU_bl','NPIKTOT']

ml_df_clean = df[ml_cols]
ml_df_clean = ml_df_clean.dropna(thresh=9) 
ml_df_clean = ml_df_clean.dropna()
#ml_df_clean = pd.get_dummies(ml_df_clean)

X = SS.fit_transform(ml_df_clean[['ABETA', 'PTAU', 'TAU', 'ratio_Ventricles_bl',
'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
'ratio_Fusiform_bl', 'ratio_ICV_bl', 'ratio_ABETA_bl', 'ratio_TAU_bl',
'ratio_PTAU_bl']])
y = ml_df_clean.sleep_deprived

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

print(pd.crosstab(index = y_train,
                  columns = y_pred_train))


print(score)

#%%
rus = RandomUnderSampler(random_state=432)
X_undersampled, y_unsampled = rus.fit_resample(ml_df_clean, ml_df_clean['DIAG_AD'])
X_undersampled.to_csv('ml_df_clean_undersampled.csv')


ml_df_clean = pd.read_csv('ml_df_clean_undersampled.csv')
# X = SS.fit_transform(ml_df_clean[[ 'ratio_Ventricles_bl',
# 'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
# 'ratio_Fusiform_bl', 'ratio_ICV_bl','insomnia']])
# y = y = ml_df_clean.DIAG_AD



def random_tree(n, extra_criteria):
    
    X = SS.fit_transform(ml_df_clean[[ 'ratio_Ventricles_bl',
    'ratio_Hippocampus_bl', 'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl',
    'ratio_Fusiform_bl', 'ratio_ICV_bl']+extra_criteria])
    y = y = ml_df_clean.DIAG_AD
    clf = RandomForestClassifier(n_estimators = 2**n, max_depth=5, random_state=0)
    clf.fit(X, y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_val_score(clf, X, y, cv=kf)
    print(f'Number of trees: {2**n},{cv_results.mean()}')
    return cv_results.mean()


model_scores = dict()
for i in range(1,):
    model_scores.update({2**i : random_tree(i, [])})
    
model_scores_insomnia = dict()
for i in range(1,6):
    model_scores_insomnia.update({2**i : random_tree(i, ['insomnia'])})

    
model_scores_OSA = dict()
for i in range(1,6):
    model_scores_OSA.update({2**i : random_tree(i, ['OSA'])})

model_scores_NPI = dict()
for i in range(1,6):
    model_scores_NPI.update({2**i : random_tree(i, ['NPIKTOT'])})


model_scores_all = dict()
for i in range(1,6):
    model_scores_all.update({2**i : random_tree(i, ['NPIKTOT', 'OSA', 'insomnia'])})  
    
#%% feature importance for model with all variables 
n = 40
res = usampling_scale_data(ml_df_clean,drop_lst,target)  
lst = ['insomnia', 'OSA', 'ratio_Ventricles_bl', 'ratio_Hippocampus_bl',
       'ratio_WholeBrain_bl', 'ratio_Entorhinal_bl', 'ratio_Fusiform_bl',
       'ratio_ICV_bl', 'NPIKTOT']
X_ = res[0]  # unscaled input
y_ = res[3]
clf = RandomForestClassifier(n_estimators =n, random_state = 5862)
title_label = '{}-fold crossvalidation random forest ({} trees)'.format(k,n)
feature_importance(X_,y_,clf,k,title_label)
plt.suptitle('Target: AD 798; not AD 798')

# plot the diagnosis distribution 
df_new = X_.copy()
df_new[target] = y_
f,axes = plt.subplots(nrows = 3,ncols=3,figsize=(18,12))
axes = axes.ravel()


for i in range(len(lst)):
    ax = sns.boxplot(data=df_new, y=lst[i],x= target,ax = axes[i])
    ax.set(xlabel='Diagnosis AD')

#%%
print(f'best score without sleep: Trees: {max(model_scores, key=model_scores.get)}, score: {model_scores[max(model_scores, key=model_scores.get)]}')
print(f'best score OSA: Trees: {max(model_scores_OSA, key=model_scores_OSA.get)}, score: {model_scores_OSA[max(model_scores_OSA, key=model_scores.get)]}')
print(f'best score insomnia: Trees: {max(model_scores_insomnia, key=model_scores_insomnia.get)}, score: {model_scores_insomnia[max(model_scores_insomnia, key=model_scores.get)]}')
print(f'best score NPIK: Trees: {max(model_scores_NPI, key=model_scores_NPI.get)}, score: {model_scores_NPI[max(model_scores_NPI, key=model_scores.get)]}')
print(f'best score all variables: Trees: {max(model_scores_all, key=model_scores_all.get)}, score: {model_scores_all[max(model_scores_all, key=model_scores.get)]}')
