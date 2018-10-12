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

df=pd.DataFrame(columns=)


dataset.f6.combine(dataset.f10, lambda x1, x2: x1 if x1 < x2 else x2)






