#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:11:35 2018

@author: adriano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#IMPORTANDO DATABASE
dataset = pd.read_csv('avila-tr.txt')

#DETECÇÃO DE DADOS AUSENTES
count_dados_ausentes = dataset.isnull().sum()

#MAPA DE CALOR 'CORRELAÇÃO'
correlacao=dataset.corr().abs().unstack().sort_values(kind='quicksort')
sns.heatmap(dataset.corr())
print(correlacao)



#COMBINANDO DUAS COLUNAS EM UMA
dataset['f6_f10']=dataset.f6.combine(dataset.f10, lambda x1, x2: x1 if x1 < x2 else x2)



#COMBINANDO DUAS COLUNAS EM UMA
dataset['f6_f10']=dataset.f6.combine(dataset.f10, lambda x1, x2: (x1+x2)/2)

#APAGANDO COLUNAS PREVIAMENTE COMBINADAS
dataset.drop(['f6'],axis=1, inplace=True)
dataset.drop(['f10'],axis=1, inplace=True)



correlacao=dataset.corr().abs().unstack().sort_values(kind='quicksort')
sns.heatmap(dataset.corr())
print(correlacao)

#REMOÇÃO DE OUTLIERS
dataset=dataset.ix[dataset['f1'] <= 11]
dataset=dataset.ix[dataset['f2'] <= 40]

#APAGANDO INSTANCIAS REPETIDAS
dataset = dataset.drop_duplicates(None, 'first')

#HISTOGRAM SEM BALANCEAMENTO
hist_dataset = dataset['class']
hist_dataset = pd.Categorical(hist_dataset).codes
sns.distplot(hist_dataset)



#BALANCEAMENTO

dataset['class'].value_counts()

from sklearn.utils import resample
X_majority = dataset[dataset['class'] == 'G']
X_B = dataset[dataset['class'] == 'B']
X_C = dataset[dataset['class'] == 'C']
X_D = dataset[dataset['class'] == 'D']
X_E = dataset[dataset['class'] == 'E']
X_W = dataset[dataset['class'] == 'W']
X_F = dataset[dataset['class'] == 'F']
X_A = dataset[dataset['class'] == 'A']
X_X = dataset[dataset['class'] == 'X']
X_Y = dataset[dataset['class'] == 'Y']
X_I = dataset[dataset['class'] == 'I']

X_majority_upsampled = resample(X_B, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123) # reproducible results

X_majority_upsampled = pd.concat([X_majority_upsampled,
                                  resample(X_C, 
                                 replace=True,
                                 n_samples=len(X_majority),
                                 random_state=123) ,
                                resample(X_D, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123), # reproducible results
                                resample(X_E, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123),# reproducible results
                                resample(X_W, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123), # reproducible results
                                resample(X_F, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123), # reproducible results
                                resample(X_A, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123), # reproducible results
                                resample(X_X, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123), # reproducible results
                                resample(X_Y, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123), # reproducible results
                                resample(X_I, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123), # reproducible results
                                ])
X_majority_upsampled['class'].value_counts()
                            
X_majority = dataset[dataset['class'] == 'H']
X_minority = dataset[dataset['class'] == 'G']

X_minority_upsampled = resample(X_majority, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_minority),     # to match minority class
                                 random_state=123) # reproducible results

X_upsampled = pd.concat([X_majority_upsampled, X_minority_upsampled, X_minority])

X_upsampled['class'].value_counts()


#HISTOGRAM COM BALANCEAMENTO
hist_dataset = X_upsampled['class']
hist_dataset = pd.Categorical(hist_dataset).codes
sns.distplot(hist_dataset)


X = X_upsampled
Y = X.iloc[:,8]
X = X.drop(['class'], axis = 1)

#DEFININDO TESTE E TREINO
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(X_train)

#DISCRETIZAÇÃO DA DATABASE
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#REDE NEURAL

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='tanh',solver='lbfgs',hidden_layer_sizes=(20,20))

clf.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

y_pred=clf.predict(X_test)

#MATRIZ DE CONFUSÃO
matriz_de_confusao = confusion_matrix(y_test, y_pred)
print(matriz_de_confusao)  

# PRECISÃO//RECALL'SENSIBILIDADE E ESPECIFICIDADE'//F1-SCORE//SUPPORT
print(classification_report(y_test, y_pred))  

#ACURÁCIA
print('Acurácia:',accuracy_score(y_test, y_pred))




######################################
dataset.to_csv('new_csv2.txt',sep='\t')




