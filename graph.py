import matplotlib.pyplot as plt 
import pandas as pd

# Leer el archivo CSV
data = pd.read_csv('results2.csv')


# Graficar las columnas del DataFrame
plt.plot(data['sigma'], data['lse'], label='LSE', marker='o')
plt.plot(data['sigma'], data['tikhonov'], label='Tikhonov', marker='x')
plt.plot(data['sigma'], data['nsga2'], label='NSGA-II', marker='s')

# Añadir etiquetas y título
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Average RMSE')
plt.xlabel(r'$\sigma_w^2$')
plt.title('Comparison of Methods')

# Añadir la leyenda
plt.legend()

# Guardar la gráfica en un archivo
plt.savefig('img/results2.png')

