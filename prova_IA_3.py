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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.metrics import r2_score, mean_absolute_error
import math
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
scaler.fit(X_test)

#X_train = scaler.transform(X_train)  

#X_test = scaler.transform(X_test)  


'''
#KNN
from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=3,algorithm='auto',weights='distance',p=2)

neigh.fit(X_train, y_train) 

y_pred = neigh.predict(X_test)
 

from sklearn.metrics import r2_score, mean_squared_error

r2_score(y_test,y_pred)

mean_absolute_error(y_test, y_pred)  

'''

from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(activation='relu',solver='adam',hidden_layer_sizes=(200,200),beta_1=0.7)
               
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

mean_absolute_error(y_test, y_pred)



'''
#############################################
dataset_train['id']=dataset_train.index + 1
dataset_train.to_csv('data_train.txt',sep='\t')
'''