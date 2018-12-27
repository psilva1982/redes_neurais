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

def sigmoidDerivada(sig):
    return sig * (1 - sig)

# Teste de função sigmoid e sigmoidDerivada
#a = sigmoid(0.5)
#b = sigmoidDerivada(a)

entradas = np.array([
    [0,0], [0,1], [1,0],[1,1]
])

saidas = np.array([
    [0], [1], [1], [0]
])

# Entrada para camada oculta
#pesos_nivel1 = np.array([
#    [-0.424, -0.740, -0.961],
#    [0.358, -0.577, -0.469]
#])

pesos_nivel1 = 2 * (np.random.random((2,3)) -1)
    
# Camada oculta para saída
#pesos_nivel2 = np.array([
#    [-0.017], [-0.893], [0.148]
#])
    
pesos_nivel2 = 2 * (np.random.random((3,1)) -1)

# O ajuste de pesos será feito conforme a variável época
# Quantas rodadas serão feitas o ajuste de pesos
epocas = 1000000
taxaAprendizagem = 0.6 
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    
    # Feed Forward    
    somaSinapse0 = np.dot(camadaEntrada, pesos_nivel1)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos_nivel2)
    camadaSaida = sigmoid(somaSinapse1)
    
    # Cálculo do erro 
    # erro = resultado_esperado - resultado_obtido
    erroCamadaSaida = saidas - camadaSaida

    # % de acerto de uma rede neural é 100 - mediaAbsolutaErro
    mediaAbsolutaErro = np.mean(np.abs(erroCamadaSaida))
    print("Erro: {}" .format(mediaAbsolutaErro))
    
    
    # Calculo dos pesos e erros
    
    ### Algoritmos de ajuste de pesos 
    
    # - Gradiente (descent gradient) - Calcula do declive da curva 
    # com derivadas parciais. As derivadas são utilizadas para 
    # saber qual direção o gradiente deve descer. Com o resultado
    # da derivada efetua o cálculo do Delta 
    
    # Calculo da derivada e delta da Camada de Saída
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    # Calculo da derivada e delta para Camada Oculta
    
    # Matriz transposta de pesos_nivel2
    pesos_nivel2_Transposta = pesos_nivel2.T
    
    deltaSaida_X_peso = deltaSaida.dot(pesos_nivel2_Transposta)
    deltaCamadaOculta = deltaSaida_X_peso * sigmoidDerivada(camadaOculta)
    
    # Back propagation 
    camadaOcultaTransposta = camadaOculta.T
    pesos_nivel2_novo = camadaOcultaTransposta.dot(deltaSaida)
    pesos_nivel2 = (pesos_nivel2 * momento) + (pesos_nivel2_novo * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesos_nivel1_novo = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos_nivel1 = (pesos_nivel1 * momento) + (pesos_nivel1_novo * taxaAprendizagem)
    
    
    
    