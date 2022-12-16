#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[1]:


## models(df,drop_lst,target)

def usampling_split_scale_data(df,drop_lst,target):
    '''
    undersampling data, split data (4:1),data scaling, pca components which could explains 90% data
    ----------------------------------
    df: the full dataframe
    target: the target feature name
    -----------
    Outputs: X_train,X_test,X_train_scaled,X_test_scaled,X_train_pca,X_test_pca,y_train,y_test
       '''
    # undersampling 
    X = df.copy()
    y = X[target]
    rus = RandomUnderSampler(random_state=432)
    X_undersampled, y_unsampled = rus.fit_resample(X, y)
    print('After undersampling data size is',len(X_undersampled),'; Resampled dataset shape %s' % Counter(y_unsampled))
    # split data
    train, test = train_test_split(X_undersampled,test_size=0.2)   
    X_train = train.drop(drop_lst,axis=1)
    y_train = train[target]
    X_test = test.drop(drop_lst,axis=1)
    y_test = test[target]
    # data scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
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
    return X_train,X_test,X_train_scaled,X_test_scaled,X_train_pca,X_test_pca,y_train,y_test,X_labels  

def models(df,drop_lst,target):
    '''
    for splitted data
    '''
    res = usampling_split_scale_data(df,drop_lst,target)
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
            
            
            
            

##### cv_models(df,drop_lst,target,k)

def usampling_scale_data(df,drop_lst,target):
    '''
    undersampling data, NOT SPLIT data (later use CROSS VALIDATION),data scaling, pca components which could explains 90% data
    ----------------------------------
    df: the full dataframe
    drop_lst: drop the features which are not gonna be used in modeling, e.g. RID,...
    target: the target feature name
    -----------
    Outputs: X,X_scaled,X_pca,y,X_labels
       '''
    # undersampling 
    y_output = df[target]
    rus = RandomUnderSampler(random_state=432)
    X_undersampled, y_unsampled = rus.fit_resample(df, y_output)
    print('After undersampling data size is',len(X_undersampled),'; Resampled dataset shape %s' % Counter(y_unsampled)) 
    # feature list for the X 
    # normal input output data
    X = X_undersampled.drop(drop_lst,axis=1)
    y = X_undersampled[target]
    # data scaling
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    # pca components
    pca = PCA(n_components=X_scaled.shape[1]) # keep all n principal components 
    pca.fit(X_scaled) # fit PCA model with scaled data
    X_pca = pca.transform(X_scaled)  #transform data onto the first two principal components
    ex_ratio = pca.explained_variance_ratio_
    cum_sum = 0
    for i in range(len(ex_ratio)):
        cum_sum += ex_ratio[i]
        if cum_sum >= 0.9:  # if it could explain 90% of the data, then stop
            break
    n_com = i         
         # PCA with first n_com components
    pca = PCA(n_com)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
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
    #print('Output dataframes sequence: X_,X_scaled,X_pca,y_')
    X_labels = ['original dataset','scaled dataset','%s pca-components'%n_com]
    return X,X_scaled,X_pca,y,X_labels 

def cv_models(df,drop_lst,target,k):
    '''
    df: full dataframe.
    drop_lst: drop the features which are not gonna be used in modeling, e.g. RID,...
    target: the target feature name
    k: folds of cross-validation
    '''
    res = usampling_scale_data(df,drop_lst,target) # Output dataframes sequence: X_,X_scaled,X_pca,y_
    
    y = res[3]
    X_labels = res[4]
    for i in range(3):
        X = res[i]
        print('- Using {}:'.format(X_labels[i]))
        # logistic regression
        C_lst = [0.001,0.01,0.1,1,10,100,1000]
        print('    - Logistic regression')
        for i in range(len(C_lst)):
            print('       - C = {}'.format(C_lst[i]))
            logreg = LogisticRegression(C=C_lst[i],solver='lbfgs',multi_class='auto',penalty='l2',max_iter=10000)
            print('          - lbfgs_L2, average weighted f1-score of {}-cross validation:{:.3f}'.format(k,cross_val_score(logreg, X, y, cv = k,scoring='f1_weighted').mean()))
            logreg = LogisticRegression(C=C_lst[i],solver='saga',multi_class='auto',penalty='l1',max_iter=10000)
            print('          - saga_L1, average weighted f1-score of {}-cross validation:{:.3f}'.format(k,cross_val_score(logreg, X, y, cv = k,scoring='f1_weighted').mean()))
            logreg = LogisticRegression(C=C_lst[i],solver='newton-cg',multi_class='auto',penalty='l2',max_iter=10000)
            print('          - newton-cg_L2, average weighted f1-score of {}-cross validation:{:.3f}'.format(k,cross_val_score(logreg, X, y, cv = k,scoring='f1_weighted').mean()))

        # decision tree
        print('    - Decision tree')
        for i in range(1,15):
            dtree = DecisionTreeClassifier(random_state=0,max_depth=i,criterion='gini')
            print('          - tree depth: {:.3f}. average weighted f1-score of {}-cross validation:{:.3f}'
                .format(i,k,cross_val_score(dtree, X, y, cv = k,scoring='f1_weighted').mean()))

        # random forest
        print('    - Random forest')
        for i in range(1,20):   
            m= 5*i
            forest = RandomForestClassifier(n_estimators=m,random_state=5862)
            print('          - {}trees. average weighted f1-score of {}-cross validation:{:.3f}'
                .format(m,k,cross_val_score(forest, X, y, cv = k,scoring='f1_weighted').mean()))
        # MLP 
        print('    - MLP')
        hls = [[50,50],[20,20]] # hidden layer size  ,[100,100]
        for i in range(len(hls)):
            mlp = MLPClassifier(solver='lbfgs',random_state=460,hidden_layer_sizes = hls[i],max_iter=20000)
            print('          - hidden layer size{}. average weighted f1-score of {}-cross validation:{:.3f}'.format(hls[i],k,cross_val_score(mlp, X, y, cv = k,scoring='f1_weighted').mean()))          


# In[4]:


# feature importance check
def feature_importance(X,y,clf,k,title_label):
    '''
    check the feature importance of the selected classification (decisiontree or random forest) model.
    X: input data
    y: output data
    clf: classification model, e.g.clf = RandomForestClassifier(n_estimators = 90, random_state = 5862) 
    Return
    ------
    dataframe of raw importance info
    boxplot of importance
    '''
    output = cross_validate(clf, X, y, cv=k, scoring = 'f1_weighted', return_estimator =True)
    d = {}  # dictionary to collect all importance dataframes 
    print("Features sorted by their score for each estimator ")
    for idx,estimator in enumerate(output['estimator']):   
        feature_importances = pd.DataFrame(estimator.feature_importances_,
                                           index = X.columns,
                                            columns=["importance_%s"% (idx+1)])
        d[idx] = feature_importances  
    df = d[0]  # dataframe to concat all dataframes in d
    for i in range(1,len(d)):
        df = pd.concat([df,d[i]],axis=1)
    df['avg_importance'] = df.mean(axis=1)
    df = df.sort_values(by = ['avg_importance'], ascending = [False])
    # insert avg_importance column as first column
    df.insert(0, 'avg_importance', df.pop('avg_importance'))
    # preparation for plotting
    dff = df.T.reset_index().iloc[1:,1:] 
    # plot feature importance
    bp = dff.boxplot(rot=30,figsize=(12,6),fontsize=12)
    bp.set_ylabel('Feature Importance',fontsize=12)
    bp.set_title('Feature importance of %s'%(title_label),fontsize=18)
    return df

