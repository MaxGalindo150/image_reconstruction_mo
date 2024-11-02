import numpy as np

# Crear una matriz de tamaño 1x3
matriz_1x3 = np.array([[1, 2, 3]])

# Crear una matriz de tamaño 2x3
matriz_2x3 = np.array([[4, 5, 6],
                       [7, 8, 9]])

# Multiplicar la matriz 1x3 por la transpuesta de la matriz 2x3
resultado = np.dot(matriz_1x3.T, matriz_2x3)

print("Matriz 1x3:")
print(matriz_1x3.shape)

print("\nMatriz 2x3:")
print(matriz_2x3.shape)

print("\nResultado de la multiplicación:")
print(resultado)