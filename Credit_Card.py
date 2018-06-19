# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 00:22:17 2018

@author: Dipankar Karmakar
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing #for using label encoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)



cr_train_data=pd.read_excel('/home/dipankar/Downloads/default of credit card clients.xls', skiprows=1)
'''At some times while reading it says UniDecodeError then use the following method(mostly happen in windows)-
cr_train_data=pd.read_csv(r'C:\Users\Dipankar-PC\Downloads\CR_data\UCI_Credit_Card.csv',encoding='utf8')
'''
'''to see the details of top 5 rows'''
print(cr_train_data.head())
'''let's have a look if there are missing or anomalous data'''
print(cr_train_data.info())

Y_cr=cr_train_data['default payment next month']
cr_train_data=cr_train_data.drop(['default payment next month','ID','SEX','PAY_5','MARRIAGE','EDUCATION','PAY_4','PAY_6','PAY_3','PAY_2','PAY_AMT5','PAY_AMT4'],axis=1)
print(cr_train_data.head())
'''split of data into train and test'''
print(Y_cr.isnull().sum().sum())
X_train,X_test,Y_train,Y_test=train_test_split(cr_train_data,Y_cr,test_size=0.2)

'''standardisation and normalisation'''
scl=StandardScaler()
data_scl=scl.fit_transform(X_train)
data_pd=pd.DataFrame(data_scl,columns=X_train.columns)

scl=StandardScaler()
data_scl1=scl.fit_transform(X_test)
data_pd1=pd.DataFrame(data_scl1,columns=X_test.columns)

print(data_pd.head())
print(data_pd1.head())


'''feature extraction'''

random_imp=RandomForestClassifier()
random_imp.fit(data_pd,Y_train)
print(random_imp.feature_importances_)
print(data_pd.columns)
a=data_pd.columns
b=random_imp.feature_importances_
c={}
for i,j in zip(a,b):
    c.update({i:j})
print(c)
'''sorting dictionary by its value'''
from collections import Counter
c1=Counter(c)
print(c1.most_common())


#=========================================================================================

'''accuracy score of random forest algorithm before pca'''

rfe=RandomForestClassifier()
rfe.fit(data_pd,Y_train)
#print(rfe.score(data_pd1,Y_test))

y_pred=rfe.predict(data_pd1)
from sklearn.metrics import accuracy_score
acc_sc=accuracy_score(Y_test,y_pred)
print(acc_sc,'-Accuracy with random forest using feature extraction before pca')

'''accuracy score of svc algorithm before pca'''

#0.825333333333-accuracy
svee=SVC()
svee.fit(data_pd,Y_train)
y_pred1=svee.predict(data_pd1)
from sklearn.metrics import accuracy_score
acc_sc1=accuracy_score(Y_test,y_pred1)
print(acc_sc1,'-Accuracy with svc using feature extraction before pca')

'''accuracy score of grb algorithm before pca'''


grb=GradientBoostingClassifier()   
grb.fit(data_pd,Y_train)
y_pred5=grb.predict(data_pd1)
from sklearn.metrics import accuracy_score
acc_sc7=accuracy_score(Y_test,y_pred5)
print(acc_sc7,'-Accuracy with gradient boosting classifier using feature extraction before pca')

'''accuracy score of abc algorithm before pca'''

abc1=AdaBoostClassifier()
abc1.fit(data_pd,Y_train)
y_pred9=abc1.predict(data_pd1)
acc_sc9=accuracy_score(Y_test,y_pred9)
print(acc_sc9,'-Accuracy with ada boosting classifier using feature extraction before pca')

'''accuracy score of etc algorithm before pca'''

etc=ExtraTreesClassifier()
etc.fit(data_pd,Y_train)
y_pred10=etc.predict(data_pd1)
acc_sc10=accuracy_score(Y_test,y_pred10)
print(acc_sc10,'-Accuracy with extra tress classifier using feature extraction before pca')

#=================================================================================================


'''dimensionality reduction with the help of eigenvalues and eigenvectors'''
cor_matt=data_pd.corr()
eig_vals, eig_vecs = np.linalg.eig(cor_matt)
#print(eig_vals)
#print('sdaddddddddddddddd')
#print(eig_vecs)
'''fiting and transforming pca'''
pca=PCA(n_components=10)
train_features = pca.fit_transform(data_pd)
test_features = pca.transform(data_pd1)

#===================================================================================

'''Accuracy with random forest using feature extraction and pca'''

rfe1=RandomForestClassifier()
rfe1.fit(train_features,Y_train)
y_pred2=rfe1.predict(test_features)
from sklearn.metrics import accuracy_score
acc_sc2=accuracy_score(Y_test,y_pred2)
print(acc_sc2,'-Accuracy with random forest using feature extraction and pca')

'''Accuracy with svc using feature extraction and pca'''

svee1=SVC()
svee1.fit(train_features,Y_train)
y_pred3=svee1.predict(test_features)
from sklearn.metrics import accuracy_score
acc_sc5=accuracy_score(Y_test,y_pred3)
print(acc_sc5,'-Accuracy with svc using feature extraction and pca')



'''Accuracy with grb using feature extraction and pca'''

grb=GradientBoostingClassifier()   
grb.fit(train_features,Y_train)
y_pred4=grb.predict(test_features)
from sklearn.metrics import accuracy_score
acc_sc6=accuracy_score(Y_test,y_pred4)
print(acc_sc6,'-Accuracy with gradient boosting classifier using feature extraction and pca')

'''Accuracy with grb using feature extraction and pca'''

abc=AdaBoostClassifier()
abc.fit(train_features,Y_train)
y_pred7=abc.predict(test_features)
acc_sc8=accuracy_score(Y_test,y_pred7)
print(acc_sc8,'-Accuracy with ada boosting classifier using feature extraction and pca')

'''Accuracy with etc using feature extraction and pca'''

etc1=ExtraTreesClassifier()
etc1.fit(train_features,Y_train)
y_pred11=etc1.predict(test_features)
acc_sc11=accuracy_score(Y_test,y_pred11)
print(acc_sc11,'-Accuracy with extra tress classifier using feature extraction and pca')

#===============================================================================================










