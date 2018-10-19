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
from math import sqrt, pow
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
y_train = dataset_train.iloc[:-1,5:]

X_train.drop(0,axis=0, inplace=True)

X_train.reset_index(drop=True, inplace=True)

X_train['Cx']=y_train['Coordenada X']
X_train['Cy']=y_train['Coordenada Y']


X_test = dataset_test.iloc[:,:5]
y_test = dataset_test.iloc[:-1,5:]

X_test.drop(0,axis=0, inplace=True)

X_test.reset_index(drop=True, inplace=True)

X_test['Cx']=y_test['Coordenada X']
X_test['Cy']=y_test['Coordenada Y']

y_test = y_test.as_matrix()

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



#FLORESTA ALEATÓRIA
hipotenusa_RANDOM=[]

for x in range(0,20):
 
    neigh = RandomForestRegressor(random_state=x)
    
    neigh.fit(X_train, y_train) 
    
    y_pred = neigh.predict(X_test)
     
    mean_absolute_error(y_test, y_pred)
  
    
    hipotenusa_RANDOM.append(0)
        
    for i in range(len(y_pred)-1):
        hipotenusa_RANDOM[x] +=  sqrt( pow( (y_pred[i][0]-y_test[i][0]) ,2) + pow( (y_pred[i][1]-y_test[i][1]) ,2))
    
    
    hipotenusa_RANDOM[x] = hipotenusa_RANDOM[x]/len(y_pred) 

'''
#REDE NEURAL OK
clf = MLPRegressor(activation='relu',solver='lbfgs',hidden_layer_sizes=(25,25),random_state = 3)
               
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mean_absolute_error(y_test, y_pred)


hipotenusa = 0

for i in range(len(y_pred)-1):
    hipotenusa +=  sqrt( pow( (y_pred[i][0]-y_test[i][0]) ,2) + pow( (y_pred[i][1]-y_test[i][1]) ,2))


hipotenusa = hipotenusa/len(y_pred) 
'''



#REDE NEURAL OK
hipotenusa=[]

for x in range(0,20):
    clf = MLPRegressor(activation='relu',solver='lbfgs',hidden_layer_sizes=(25,25),random_state = x)
               
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    mean_absolute_error(y_test, y_pred)
    
    hipotenusa.append(0)
    
    for i in range(len(y_pred)-1):
        hipotenusa[x] +=  sqrt( pow( (y_pred[i][0]-y_test[i][0]) ,2) + pow( (y_pred[i][1]-y_test[i][1]) ,2))


    hipotenusa[x] = hipotenusa[x]/len(y_pred) 




'''
#############################################
dataset_train['id']=dataset_train.index + 1
dataset_train.to_csv('data_train.txt',sep='\t')
'''