#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.tree import export_text
import mglearn
from dashboard_one import *
from feature_selection import *
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate


# In[5]:


## models_os(df,drop_lst,target)

def oversampling_split_scale_data(df,drop_lst,target):
    '''
    oversampling data, split data (4:1),data scaling, pca components which could explains 90% data
    ----------------------------------
    df: the full dataframe
    target: the target feature name
    -----------
    Outputs: X_train,X_test,X_train_scaled,X_test_scaled,X_train_pca,X_test_pca,y_train,y_test
       '''    
    # split data
    drop_lst_2 = drop_lst[0:-1]
    train, test = train_test_split(df,test_size=0.2)  
    X_train = train.drop(drop_lst,axis=1)
    y_train = train[target]
    X_test = test.drop(drop_lst,axis=1)
    y_test = test[target]

    # oversampling
    ros = RandomOverSampler(random_state=432)
    X_oversampled,y_oversampled = ros.fit_resample(X_train,y_train)
    print('After oversampling, train data size is',len(X_oversampled),'; Resampled dataset shape %s' % Counter(y_oversampled))
    
    # data scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_oversampled)
    X_test_scaled = scaler.transform(X_test)
    # pca components
    pca = PCA(n_components=X_train_scaled.shape[1]) # keep all n principal components 
    pca.fit(X_train_scaled) # fit PCA model with scaled data
    X_pca = pca.transform(X_train_scaled)  #transform data onto the first two principal components
    ex_ratio = pca.explained_variance_ratio_
    cum_sum = 0
    for i in range(len(ex_ratio)):
        cum_sum += ex_ratio[i]
        if cum_sum >= 0.9:  # if it could explain 90% of the data, then stop
            break
    n_com = i         
         # PCA with first n_com components
    pca = PCA(n_com)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    '''   #plot 
    plt.bar(range(1,len(pca.explained_variance_ratio_ )+1),pca.explained_variance_ratio_ )
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Components')
    plt.plot(range(1,len(pca.explained_variance_ )+1),
             np.cumsum(pca.explained_variance_ratio_),
             c='red',
             label="Cumulative Explained Variance ratio")
    plt.legend(loc='upper left')'''
    print('\n{} principle components are needed to explain 90% of the data\n'.format(n_com))  
    print('Output dataframes sequence: X_train,X_test,X_train_scaled,X_test_scaled,X_train_pca,X_test_pca,y_train,y_test')
    X_labels = ['original dataset','scaled dataset','%s pca-components'%n_com]
    return X_oversampled,X_test,X_train_scaled,X_test_scaled,X_train_pca,X_test_pca,y_oversampled,y_test,X_labels  

def models_os(df,drop_lst,target):
    '''
    for splitted data
    '''
    res = oversampling_split_scale_data(df,drop_lst,target)
    y_train = res[6]
    y_test = res[7]
    X_labels = res[8]
    for i in range(len(X_labels)):
        print('- Using {}:'.format(X_labels[i]))
        X_train = res[2*i]
        X_test = res[2*i+1]
        # logistic regression
        C_lst = [0.001,0.01,0.1,1,10,100,1000]
        print('    - Logistic regression')
        for i in range(len(C_lst)):
            print('       - C = {}'.format(C_lst[i]))
            logreg = LogisticRegression(C=C_lst[i],solver='lbfgs',multi_class='auto',penalty='l2',max_iter=10000).fit(X_train,y_train)
            print('          - lbfgs_L2, Training set f1-score:{:.3f}, Test set f1-score: {:.3f}'
                  .format(f1_score(logreg.predict(X_train),y_train,average='weighted'),f1_score(logreg.predict(X_test),y_test,average='weighted')))
            logreg = LogisticRegression(C=C_lst[i],solver='saga',multi_class='auto',penalty='l1',max_iter=10000).fit(X_train,y_train)
            print('          - saga_L1, Training set f1-score:{:.3f}, Test set f1-score: {:.3f}'
                  .format(f1_score(logreg.predict(X_train),y_train,average='weighted'),f1_score(logreg.predict(X_test),y_test,average='weighted')))
            logreg = LogisticRegression(C=C_lst[i],solver='newton-cg',multi_class='auto',penalty='l2',max_iter=10000).fit(X_train,y_train)
            print('          - newton-cg_L2, Training set f1-score:{:.3f}, Test set f1-score: {:.3f}'
                  .format(f1_score(logreg.predict(X_train),y_train,average='weighted'),f1_score(logreg.predict(X_test),y_test,average='weighted')))

        # decision tree
        print('    - Decision tree')
        for i in range(1,15):
            dtree = DecisionTreeClassifier(random_state=0,max_depth=i,criterion='gini')
            dtree.fit(X_train,y_train)
            print('          - tree depth: {:.3f}. f1-score on training data: {:.3f} f1-score on test data: {:.3f}'
              .format(i,f1_score(dtree.predict(X_train),y_train,average='weighted'),f1_score(dtree.predict(X_test),y_test,average='weighted')))
        # random forest
        print('    - Random forest')
        for i in range(1,20):   
            m= 5*i
            forest = RandomForestClassifier(n_estimators=m,random_state=5862)
            forest.fit(X_train,y_train)
            print('          - {}trees. f1-score on training data: {:.3f} f1-score on test data: {:.3f}'
              .format(m,f1_score(forest.predict(X_train),y_train,average='weighted'),f1_score(forest.predict(X_test),y_test,average='weighted')))
        # MLP  
        print('    - MLP')
        hls = [[50,50],[20,20]] # hidden layer size  ,[100,100],[50,50,50]
        for i in range(len(hls)):
            mlp = MLPClassifier(solver='lbfgs',random_state=460,hidden_layer_sizes = hls[i],max_iter=20000)
            mlp.fit(X_train,y_train)
            print('          - hidden layer size{}. f1-score on training data: {:.3f} f1-score on test data: {:.3f}'.format(hls[i],f1_score(mlp.predict(X_train),y_train,average='weighted'),f1_score(mlp.predict(X_test),y_test,average='weighted'))) 
            
            


# In[ ]:




