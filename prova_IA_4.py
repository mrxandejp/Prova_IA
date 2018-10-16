    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """
    Created on Sat Oct 13 23:21:50 2018
    
    @author: adriano
    """
    
    import pandas as pd
    import seaborn as sns
    dataset = pd.read_csv("csv_result-Genes_Atividade_IA.csv")
    
    dataset.drop(['id'],axis=1, inplace=True)
    dataset.drop(['class'],axis=1, inplace=True)
    
    
    
    #PŔE PROCESSAMENTO
    
    #DETECÇÃO DE DADOS AUSENTES
    count_dados_ausentes = dataset.isnull().sum()
    
    #MAPA DE CALOR 'CORRELAÇÃO' ANTES DO MERGE
    correlacao=dataset.corr().abs().unstack().sort_values(kind='quicksort')
    sns.heatmap(dataset.corr())
    print(correlacao)
    
    #APAGANDO INSTANCIAS REPETIDAS
    dataset = dataset.drop_duplicates(None, 'first')
    
    #COMBINANDO DUAS COLUNAS EM UMA
    dataset['COM:endo_smot_mus']=dataset.endometrium.combine(dataset.smooth_muscle, lambda x1, x2: (x1+x2)/2)
    dataset['COM:lymph_appendix']=dataset.lymph_node.combine(dataset.appendix, lambda x1, x2: (x1+x2)/2)
    dataset['COM:lung_spleen']=dataset.lung.combine(dataset.spleen, lambda x1, x2: (x1+x2)/2)
    dataset['COM:lymph_appendix_tonsil']=dataset['COM:lymph_appendix'].combine(dataset.tonsil, lambda x1, x2: (x1+x2)/2)
    dataset['COM:rectum_colon']=dataset.rectum.combine(dataset.colon, lambda x1, x2: (x1+x2)/2)
    dataset['COM:bladder_gallb']=dataset.bladder.combine(dataset.gallbladder, lambda x1, x2: (x1+x2)/2)
    dataset['COM:bladder_gallb_placent']=dataset['COM:bladder_gallb'].combine(dataset.placenta, lambda x1, x2: (x1+x2)/2)
    dataset['COM:bladder_gallb_placent_adtiss']=dataset['COM:bladder_gallb_placent'].combine(dataset.adipose_tissue, lambda x1, x2: (x1+x2)/2)
    dataset['COM:bladder_gallb_placent_adtiss_lung_spleen']=dataset['COM:bladder_gallb_placent_adtiss'].combine(dataset['COM:lung_spleen'], lambda x1, x2: (x1+x2)/2)
    
    #APAGANDO CLASSES MESCLADAS
    dataset.drop(['endometrium'],axis=1, inplace=True)
    dataset.drop(['smooth_muscle'],axis=1, inplace=True)
    dataset.drop(['lymph_node'],axis=1, inplace=True)
    dataset.drop(['appendix'],axis=1, inplace=True)
    dataset.drop(['lung'],axis=1, inplace=True)
    dataset.drop(['spleen'],axis=1, inplace=True)
    dataset.drop(['tonsil'],axis=1, inplace=True)
    dataset.drop(['COM:lymph_appendix'],axis=1, inplace=True)
    dataset.drop(['rectum'],axis=1, inplace=True)
    dataset.drop(['colon'],axis=1, inplace=True)
    dataset.drop(['bladder'],axis=1, inplace=True)
    dataset.drop(['gallbladder'],axis=1, inplace=True)
    dataset.drop(['COM:bladder_gallb'],axis=1, inplace=True)
    dataset.drop(['placenta'],axis=1, inplace=True)
    dataset.drop(['COM:bladder_gallb_placent'],axis=1, inplace=True)
    dataset.drop(['adipose_tissue'],axis=1, inplace=True)
    dataset.drop(['COM:bladder_gallb_placent_adtiss'],axis=1, inplace=True)
    dataset.drop(['COM:lung_spleen'],axis=1, inplace=True)
    
    #MAPA DE CALOR 'CORRELAÇÃO' DEPOIS DO MERGE
    correlacao=dataset.corr().abs().unstack().sort_values(kind='quicksort')
    sns.heatmap(dataset.corr())
    print(correlacao)
    
    X = dataset.iloc[:,0:] 

#NORMALIZAÇÃO
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
MinMaxScaler(copy=True, feature_range=(0, 1))

X=scaler.transform(X)


###############################################################################
#                                                                             #       
#                           #Clustering Kmeans#                               # 
#                                                                             #           
###############################################################################

from sklearn.cluster import KMeans

#n_clusters = 2, 5, 10, e 100
#init = k-means++ / random / 1ª favorece a convergencia 2º randomico
#max_iter = quantidade maxima de iteracoes

kmeans = KMeans(n_clusters = 50, init = 'k-means++', max_iter=300)


#Ajustando
kmeans.fit(X)

#definindo os clusters
clusters_kmeans = kmeans.predict(X)

#tabela com os valores//respectivos clusters
table_finale_kmeans = X

table_finale_kmeans=pd.DataFrame(table_finale_kmeans)

table_finale_kmeans['clusters'] = clusters_kmeans

qtd_por_clusters_kmeans=table_finale_kmeans['clusters'].value_counts()

qtd_por_clusters_kmeans.sum()


###############################################################################
#                                                                             #       
#                      #Clustering Hierárquico#                               # 
#                                                                             #           
###############################################################################


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage 
from scipy import cluster
import numpy as np , collections
import collections

Z = linkage(X,'ward','euclidean')

#criação de  
clusters_H = cluster.hierarchy.cut_tree(Z, n_clusters=50)



table_finale_H = X

table_finale_H = pd.DataFrame(table_finale_H)

table_finale_H['clusters'] = clusters_H

qtd_por_clusters_hier=table_finale_H['clusters'].value_counts()

qtd_por_clusters_hier.sum()

###############################################################################
#                                                                             #       
#                               #GRÁFICOS#                                    # 
#                                                                             #           
###############################################################################


#HISTOGRAM COORDENADA Y
hist_clusters_kmeans = qtd_por_clusters_kmeans

hist_clusters_kmeans = pd.Categorical(hist_clusters_kmeans).codes

sns.distplot(hist_clusters_kmeans)


#HISTOGRAM COORDENADA Y
hist_clusters_hier = qtd_por_clusters_hier

hist_clusters_hier = pd.Categorical(hist_clusters_hier).codes

sns.distplot(hist_clusters_hier)



###############################################################################
qtd_por_clusters_hier['id']=qtd_por_clusters_hier.index + 1
qtd_por_clusters_hier.to_csv('NORM_new_csv4.txt',sep='\t')












