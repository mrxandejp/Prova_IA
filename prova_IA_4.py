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

#kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter=300)
#k=2 apenas 1 instancia pertencente ao cluster 1

#kmeans = KMeans(n_clusters = 5, init = 'random')
#k=5 pouca variação , clusters 1 e 2 apenas com 1 instancia

#kmeans = KMeans(n_clusters = 10, init = 'random')
#k=10 ainda com pouca variação, dados super concentrados no cluster 0,
#clusters 3 e 6, apenas com 1 instancia

kmeans = KMeans(n_clusters = 100, init = 'random')
#apresenta mais variações, aparentemente nenhum cluster dominando o outro

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



###############################################################################
#                                                                             #       
#                      #Clustering Hierárquico#                               # 
#                                                                             #           
###############################################################################


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

np.set_printoptions(precision=5, suppress=True)


Z = linkage(X, 'ward','euclidean')


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))


from scipy.cluster.hierarchy import fcluster
max_d = 50

#criação de  1300 clusters
clusters_H = fcluster(Z, max_d, criterion='distance')
clusters_H
 

table_finale_H = X
table_finale_H['cluster'] = clusters_H



















