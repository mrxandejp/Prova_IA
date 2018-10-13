#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:03:27 2018

@author: adriano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#IMPORTANDO DATABASE

dataset_train = pd.read_csv('base_treinamento.csv')

dataset_test = pd.read_csv('base_teste.csv')

#CONTANDO TODAS AS VARIAVEIS VAZIAS
count_dados_ausentes_train = dataset_train.isnull().sum()
#count_dados_ausentes_test = dataset_test.isnull().sum()

#dados duplicados
dataset_train = dataset_train.drop_duplicates(None, 'first')
#dataset_test = dataset_test.drop_duplicates(None, 'first')


X_train = dataset_train.iloc[:,:5]
y_train = dataset_train.iloc[:,5:]


X_test = dataset_test.iloc[:,:5]
y_test = dataset_test.iloc[:,5:]

#correlação
correlacao=X_train.corr().abs().unstack().sort_values(kind='quicksort')
sns.heatmap(X_train.corr())
print(correlacao)

#HISTOGRAM COORDENADA X
hist_dataset = y_train['Coordenada X']
hist_dataset = pd.Categorical(hist_dataset).codes
sns.distplot(hist_dataset)

#HISTOGRAM COORDENADA Y
hist_dataset = y_train['Coordenada Y']
hist_dataset = pd.Categorical(hist_dataset).codes
sns.distplot(hist_dataset)

#METODO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)  

'''
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

X_train, y_train = make_regression(n_samples=10, n_targets=3, random_state=1)

MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X_train, y_train).predict(X_test)

'''


'''
#KNN
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=5)

neigh.fit(X_train, y_train) 

y_pred = neigh.predict(X_test)
 

from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test,y_pred)

mean_squared_error(y_test, y_pred) 
'''



'''
from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(activation='relu',solver='adam',hidden_layer_sizes=(200,200,200,200))

clf.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_squared_error

y_pred=clf.predict(X_test)

r2_score(y_test,y_pred)

mean_squared_error(y_test, y_pred)  
'''
import math
def hypo(x,y):
    return math.hypot(x,y)


#KNN
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor



rgr=MultiOutputRegressor(MLPRegressor(activation='relu',solver='adam',hidden_layer_sizes=(200,200)))

rgr.fit(X_train, y_train)

y_pred = rgr.predict(X_test)



rgr.score(X_train,y_train)



from sklearn.metrics import r2_score, mean_absolute_error

r2_score(y_test,y_pred)

mean_absolute_error(y_test, y_pred,multioutput='uniform_average') 
'''

#############################################
dataset_train['id']=dataset_train.index + 1
dataset_train.to_csv('data_train.txt',sep='\t')