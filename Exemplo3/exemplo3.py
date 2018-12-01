'''
Exemplo 03 - Ajuste / Treinamento de pesos 
Perceptron de uma camada 
Resolve apenas problemas linearmente separáveis
'''

import numpy as np 

# Operador lógico E => 0 e 0 = 0, 1 e 0 = 0, 1 e 1 = 1
# Problema linearmente separável
# entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
# saidas = np.array([0, 0, 0, 1]) # Saida esperada

# Operador 'Ou'
# Problema linearmente separável
# entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
# saidas = np.array([0, 1, 1, 1]) # Saida esperada

# Operador XOR - disjunção exclusiva - Sem solução para esse caso 
# Problema não linearmente separável
entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([0, 1, 1, 0]) # Saida esperada

# Pesos iniciais
pesos = np.array([0.0, 0.0])

taxaAprendizagem = 0.1

# stepFunction - Função de ativação simples
def funcaoAtivacao(soma):
    if soma >= 1:
        return 1
    
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return funcaoAtivacao(s)

# Função de treinamento que ajusta os pesos das sinapses 
# Encontra os pesos corretos para saída 
def treinar():
    erroTotal = 1
    while erroTotal != 0: # Somente vai para quanto resolver todo problema (muito difícil de ocorrer no mundo real)
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro

            for j in range(len(pesos)):
                pesos[j] = (pesos[j] + taxaAprendizagem * entradas[i][j] * erro)
                print('Peso atualizado: {}' .format(pesos[j]))
        
        print('Total de erros {} ' .format(erroTotal))

print('Rede neural antes do treinamento')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
print('\nExecutando treinamento')

treinar()

print('\nRede neural após treinamento') 
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
