'''
Exemplo de Perceptron de 1 camada 
'''

entradas = [-1, 7, 5]
pesos = [0.8, 0.1, 0]

def funcao_soma(e, p):
    soma = 0 
    for i in range(3):
        soma += e[i] * p[i]
    
    return soma

resultado_soma = funcao_soma(entradas, pesos)

def funcao_ativacao(soma):
    if soma >= 1:
        return 1

    return 0

ativado = funcao_ativacao(resultado_soma)

print('Ativado >>> {}' .format(ativado))