#
# Main Script for the moment 
#
# Author: Pedro Pazos Curra

import sys
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt


# Funciones auxiliares
# Eventualmente, en sendas librerías auxiliares o lo que toque

def one_hot_encoding(entrada, indice_caracteristica = 0):

    array_entrada = np.array(entrada)
    array_salida = np.array([])
    dict_caracteristicas = {}

    for fila in array_entrada:

        caracteristica = fila[indice_caracteristica]

        if caracteristica not in dict_caracteristicas:
            # Elem no leído antes
            dict_caracteristicas[caracteristica] = dict_caracteristicas.__len__()

    num_caracteristicas = dict_caracteristicas.__len__()

    nuevo_set_columnas = np.zeros(num_caracteristicas)

    for fila in array_entrada:

        caracteristica = fila[indice_caracteristica]
        fila = np.append(nuevo_set_columnas, np.delete(fila, indice_caracteristica))
        fila[dict_caracteristicas[caracteristica]] = 1
        array_salida = [array_salida, fila]

    return array_salida

def estandarizar_de_0_a_1(valor, min, max):
    return (valor-min)/(max-min)

def estandarizar_array(array):
    min = np.min(array)
    max = np.max(array)
    array_salida = np.zeros_like(array)

    for i, elem in enumerate(array):
        array_salida[i] = estandarizar_de_0_a_1(elem, min, max)

    return array_salida

def transponer_matriz(matriz):

    forma_matriz = np.shape(matriz)

    num_filas = forma_matriz[0]
    matriz_salida = np.zeros_like(matriz).reshape(-1, num_filas)
    
    # Lista vacia -> Devuelve lo mismo
    if len(forma_matriz) == 0: matriz_salida = matriz

    # Array 1-D
    elif len(forma_matriz) == 1:
        matriz_salida = np.reshape(matriz, (-1, num_filas))

    # Matriz 2-D
    else:
        
        for i, fila in enumerate(matriz):
            for j, elem in enumerate(fila):
                if i==j:
                    matriz_salida[i][j] = elem
                else:
                    matriz_salida[j][i] = elem

    return(matriz_salida)

def separar_entrenamiento_y_testing(X, y, fraccion_test = 0.25):

    # Toma y calculo de dimensiones
    num_muestras, num_caracteristicas = np.shape(X)
    num_muestras, num_clases = np.shape(y)
    tamanho_test = math.floor(num_muestras * fraccion_test)
    tamanho_train = num_muestras - tamanho_test

    # Inicializacion de arrays de salida
    X_test = np.zeros((tamanho_test, num_caracteristicas)) 
    y_test = np.zeros((tamanho_test, num_clases))
    X_train = np.zeros((tamanho_train, num_caracteristicas))
    y_train = np.zeros((tamanho_train, num_clases))

    generador_random = np.random.default_rng()

    # Test
    indices_visitados = set()
    contador_indice = 0

    while contador_indice < tamanho_test:
        indice_actual = generador_random.integers(low=0, high=num_muestras)
        if indice_actual in indices_visitados: continue
        else:
            indices_visitados.add(indice_actual)
            X_test[contador_indice] = X[indice_actual]
            y_test[contador_indice] = y[indice_actual]
            contador_indice = contador_indice + 1

    # Train
    indices_visitados = set()
    contador_indice = 0
    while contador_indice < tamanho_train:
        indice_actual = generador_random.integers(low=0, high=num_muestras)
        if indice_actual in indices_visitados: continue
        else:
            indices_visitados.add(indice_actual)
            X_train[contador_indice] = X[indice_actual]
            y_train[contador_indice] = y[indice_actual]
            contador_indice = contador_indice + 1

    return (X_train, X_test, y_train, y_test)

# Cargar dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Agrupamos las caracteristicas para estandarizar las medidas
X_transpuesta = transponer_matriz(X)
X_estandarizada = np.zeros_like(X_transpuesta)

for i, muestras_caracteristica in enumerate(X_transpuesta):
    X_estandarizada[i] = estandarizar_array(muestras_caracteristica)

X_estandarizada = transponer_matriz(X_estandarizada)

N_CARACTERISTICAS = np.shape(X_estandarizada)[1]   # Entrada
N_CLASES = np.shape(y)[1]            # Salida

X_train, X_test, y_train, y_test = separar_entrenamiento_y_testing(X_estandarizada, y, fraccion_test = 0.20)


# Funciones de activación (https://es.wikipedia.org/wiki/Funci%C3%B3n_de_activaci%C3%B3n)
##  Las funciones de activación añaden no-linearidad en la RN. Permiten modelar relaciones complejas.
## Comúnmente, ReLU Se usa en input, Sigmoide en output

def sigmoide(z):
    return 1/(1 + math.e**(-z))

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivada(x):
    return np.where(x > 0, 1, 0)

def sigmoide_derivada(z):
    return z * (1 - z)

################################ Linea del desconocimiento ################################


# Retropropagación
##  Paso 1: Propagación de input hacia adelante, recibiendo output
##  Paso 2: Cálculo de pérdida dada salida esperada
##  Paso 3: Propagación hacia atrás, se calculan las gradientes de pérdida de cada peso usando regla de la cadena. Las gradientes indican cómo recalcular el peso
##  Paso 4: Recalcular el peso con algoritmo que reduzca pérdida (típicamente, con algoritmo de optimización como Descenso de Gradiente Estocástico) 

capa_entrada = np.zeros(N_CARACTERISTICAS)
capa_salida = np.zeros(N_CLASES)