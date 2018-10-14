#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:21:50 2018

@author: adriano
"""

import pandas as pd
dataset = pd.read_csv("csv_result-Genes_Atividade_IA.csv")

X = dataset.iloc[:,0:33] 
X.drop(['id'],axis=1, inplace=True)


from sklearn.cluster import KMeans

#n_clusters = 2, 5, 10, e 100
#init = k-means++ / random / 1ª favorece a convergencia 2º randomico
#max_iter = quantidade maxima de iteracoes
kmeans = KMeans(n_clusters = 2, init = 'k-means++')
#kmeans_5 = KMeans(n_clusters = 5, init = 'random')
#kmeans_10 = KMeans(n_clusters = 10, init = 'random')
#kmeans_100 = KMeans(n_clusters = 100, init = 'random')

##################### k = 2 ###############################
#Ajustando
kmeans.fit(X)

#Mostrando Centroides
centroides = kmeans.cluster_centers_

#distancia de cada instancia a cada cluster
distance = kmeans.fit_transform(X)

#Codigo do cluster para cada instancia
labels = kmeans.labels_

#definindo os clusters
clusters = kmeans.predict(X)

#tabela com os valores//respectivos clusters

table_finale = X
table_finale['cluster'] = clusters





























