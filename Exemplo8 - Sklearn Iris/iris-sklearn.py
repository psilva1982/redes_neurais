# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 02:35:17 2018

@author: paulo.csilva
"""

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

# Flor iris
iris = datasets.load_iris()

entradas = iris.data
saidas = iris.target

redeNeural = MLPClassifier(
            verbose=True, 
            max_iter=1000, 
            tol=0.00001,
            learning_rate_init=0.001,
            )

redeNeural.fit(entradas, saidas)

# Efetua a classificação de uma entrada específica
redeNeural.predict([[5, 7.2, 5.1, 2.2]])