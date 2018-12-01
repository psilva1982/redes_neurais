'''
Exemplo de rede neural multi-camada 
com função de ativação sigmoide 

Serve para problemas não linearmente separáveis
Exemplo resolvendo problema XOR
'''

import numpy as np

# Não retorna valores negativos
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

entradas = np.array([
    [0,0], [0,1], [1,0],[1,1]
])

saidas = np.array([
    [0], [1], [1], [0]
])

# Entrada para camada oculta
pesos_nivel1 = np.array([
    [-0.424, -0.740, -0.961],
    [0.358, -0.577, -0.469]
])

# Camada oculta para saída
pesos_nivel2 = np.array([
    [-0.017], [-0.893], [0.148]
])

# O ajuste de pesos será feito conforme a variável época
# Quantas rodadas serão feitas o ajuste de pesos
epocas = 100

# Feed Forward
for j in range(epocas):
    camadaEntrada = entradas
    
    somaSinapse0 = np.dot(camadaEntrada, pesos_nivel1)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos_nivel2)
    camadaSaida = sigmoid(somaSinapse1)

''' 
Continuar em cálculo de erro 
Aula 27 
'''

