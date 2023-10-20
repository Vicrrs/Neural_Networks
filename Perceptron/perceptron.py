# Problema de Função lógica
import numpy as np

# Funcao de ativacao degrau unitario
def step_function(soma):
    if soma >= 1:
        return 1
    return 0

# Dados de treino (entradas e saidas esperadas)
entradas = ([
        [0,0],
        [0,1],
        [1,0],
        [1,1]])
saidas = np.array([0,0,0,1])

# Inicializando os pesos (2 entradas + bias)
pesos = np.array([0.0, 0.0])
taxa_aprendizado = 0.1

# Treinamento
for epoca in range(100):
    erro_total = 0
    for i in range(len(saidas)):
        soma = np.dot(entradas[i], pesos)
        saida_calculada = step_function(soma)
        erro = saidas[i] - saida_calculada
        erro_total += erro
        for j in range(len(pesos)):
            pesos[j] = pesos[j] + (taxa_aprendizado * entradas[i][j] * erro)
    if erro_total == 0:
        break
    
    
    
def predizer(entrada):
    s = np.dot(entrada, pesos)
    return step_function(s)

print(predizer([0, 1]))  # Deve retornar 0
print(predizer([1, 1]))  # Deve retornar 1


