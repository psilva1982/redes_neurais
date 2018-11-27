'''
Exemplo de Perceptron de 1 camada 
utilizando numpy 
'''

import numpy as np 

entradas = np.array([-1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def funcao_soma(e, p):
    return e.dot(p)

resultado_soma = funcao_soma(entradas, pesos)

def funcao_ativacao(soma):
    if soma >= 1:
        return 1

    return 0

ativado = funcao_ativacao(resultado_soma)

print('Ativado >>> {}' .format(ativado))

