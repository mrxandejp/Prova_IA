#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 22:57:49 2018

@author: adriano & mrxandejp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#IMPORTANDO DATABASE
dataset = pd.read_csv('csv_result-Autism-Adult-Data.csv')

#SUBSTITUINDO '?' POR VAZIO
def clean_text(text):
    #text=text.lower()
    text = text.replace("?",np.nan)
    return text

dataset=clean_text(dataset)

#CONTANDO TODAS AS VARIAVEIS VAZIAS
count_dados_ausentes = dataset.isnull().sum()

#APAGANDO COLUNAS DISPENSAVEIS
dataset.drop(['age_desc'],axis=1, inplace=True)
dataset.drop(['relation'],axis=1, inplace=True)
dataset.drop(['id'],axis=1, inplace=True)
dataset.drop(['used_app_before'],axis=1, inplace=True)

#SUBSTITUINDO VALORES AUSENTES PELA MODA
dataset['ethnicity'].fillna(dataset['ethnicity'].mode()[0],inplace=True)

#APAGANDO RUIDO
dataset.drop(dataset[dataset.age=='383'].index ,inplace=True)

#APAGANDO LINHAS QUE POSSUEM INSTANCIAS AUSENTE
dataset.drop(62,inplace=True)
dataset.drop(91,inplace=True)

dataset = dataset.drop_duplicates(None, 'first')

dataset['austim'].value_counts()


#CLASSES DESBALANCEADAS
hist_dataset = dataset['austim']
hist_dataset = pd.Categorical(hist_dataset).codes
sns.distplot(hist_dataset,kde=False,bins=3)

#BALANCEAMENTO DAS CLASSES
from sklearn.utils import resample
X_majority = dataset[dataset.austim == 'no']
X_minority = dataset[dataset.austim == 'yes']

X_majority_upsampled = resample(X_minority, 
                                 replace=True,    # sample without replacement
                                 n_samples=len(X_majority),     # to match minority class
                                 random_state=123) # reproducible results

X_downsampled = pd.concat([X_majority_upsampled, X_majority])

X_downsampled.austim.value_counts()

#CLASSES BALANCEADAS
hist_dataset = X_downsampled['austim']
hist_dataset = pd.Categorical(hist_dataset).codes
sns.distplot(hist_dataset,kde=False,bins=3)


#DISCRETIZAÇÃO DA DATABASE
X = X_downsampled
Y = X.iloc[:,14]
X = X.drop(['austim'], axis = 1)

X['gender'] = pd.Categorical(X['gender']).codes
X['ethnicity'] = pd.Categorical(X['ethnicity']).codes
X['jundice'] = pd.Categorical(X['jundice']).codes
X['contry_of_res'] = pd.Categorical(X['contry_of_res']).codes
X['Class/ASD'] = pd.Categorical(X['Class/ASD']).codes


#MAPA DE CALOR 'CORRELAÇÃO'
X.corr()
sns.heatmap(X.corr())

X.drop(['result'],axis=1, inplace=True)

X.corr()
sns.heatmap(X.corr())

###############################################################################

#PRIMEIRO MÉTODO 'KNN'
print('=================KNN=================')
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

from sklearn.neighbors import KNeighborsClassifier 
 
classifier = KNeighborsClassifier(n_neighbors=1)  

classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  

#MATRIZ DE CONFUSÃO
cm=confusion_matrix(y_test,y_pred)
print(cm)  

# PRECISÃO//RECALL'SENSIBILIDADE E ESPECIFICIDADE'//F1-SCORE//SUPPORT
print(classification_report(y_test, y_pred))  

#ACURÁCIA
print('Acurácia:',accuracy_score(y_test, y_pred))

#sensibilidade
sensibilidade= cm[0,0]/(cm[0,1]+cm[0,0])
print('Sensibilidade:',sensibilidade)

#especificidade
especificidade= cm[1,1]/(cm[1,1]+cm[1,0])
print('Especificidade:',especificidade)


###############################################################################

#SEGUNDO MÉTODO 'SVM'
print('=================SVM=================')
from sklearn import svm

classifier_SVM = svm.SVC(kernel='poly',C=3.0, degree = 6,coef0=0.0)  

classifier_SVM.fit(X_train, y_train)  

y_pred_SVM = classifier_SVM.predict(X_test) 


#MATRIZ DE CONFUSÃO
matriz_de_confusao = confusion_matrix(y_test, y_pred_SVM)
print(matriz_de_confusao)  

# PRECISÃO//RECALL'SENSIBILIDADE E ESPECIFICIDADE'//F1-SCORE//SUPPORT
print(classification_report(y_test, y_pred_SVM))  

#ACURÁCIA
print('Acurácia:',accuracy_score(y_test, y_pred_SVM))

#sensibilidade
sensibilidade= matriz_de_confusao[0,0]/(matriz_de_confusao[0,1]+matriz_de_confusao[0,0])
print('Sensibilidade:',sensibilidade)

#especificidade
especificidade= matriz_de_confusao[1,1]/(matriz_de_confusao[1,1]+matriz_de_confusao[1,0])
print('Especificidade:',especificidade)



###############################################################################
#Y['austim'] = pd.Categorical(Y['austim']).codes
Y.to_csv('saida',sep='\t')

#GRAFICO DE CORRELAÇAO
dataset.plot(x='A5_Score',y='result',kind='scatter',color='y')
