import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SSM')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from utils import nextpow2

import pickle
from types import SimpleNamespace
from Generate_Standard_Input_Signal import generate_standard_input_signal
from Generate_SSM_Model import generate_ssm_model
from Generate_Linear_Model import generate_linear_model
from mu_from_d import mu_from_d
from Generate_Measurements import generate_measurements
from skimage.draw import polygon, disk

# Definir dimensiones
Nz = 100  # Resolución en profundidad
Nx = 100  # Resolución espacial a lo largo de la dirección de escaneo
Nt = 100
c0 = 1
sigma = 1
fsize = 12  # Tamaño de fuente para las etiquetas de las figuras

# Crear matriz W
W = np.zeros((Nz, Nx))

# Definir posiciones fijas para las grietas
px = [20, 40, 60, 80]  # Coordenadas x fijas
py = [30, 50, 70, 90]  # Coordenadas y fijas

# Crear las estructuras de las grietas usando diferentes características de forma
# Punto
s = 2  # Tamaño del punto
rr, cc = polygon([py[0], py[0], py[0] + s, py[0] + s], [px[0], px[0] + s, px[0] + s, px[0]])
W[rr, cc] = 1

# Triángulo
s = 13  # Longitud de los lados del triángulo
rr, cc = polygon([py[1], py[1] + s, py[1]], [px[1], px[1], px[1] + s])
W[rr, cc] = 0.75

# Rectángulo
s = 9  # Tamaño del rectángulo
rr, cc = polygon([py[2], py[2], py[2] + s, px[2] + s], [px[2], px[2] + s, px[2] + s, px[2]])
W[rr, cc] = 0.5

# Círculo
s = 9  # Radio del círculo
rr, cc = disk((py[3], px[3]), s, shape=W.shape)
W[rr, cc] = 0.25

# Mostrar la sección transversal de la sonda
plt.imshow(W.T, aspect='auto', cmap='gray')
plt.xlabel('scanning direction x', fontsize=fsize)
plt.ylabel('depth z', fontsize=fsize)
plt.title('Cross section of the probe', fontsize=fsize)
plt.grid(True)
plt.colorbar()
plt.gca().invert_yaxis()  # Equivalente a 'axis xy' en MATLAB
plt.axis('image')  # Ajusta la relación de aspecto
plt.savefig('img/2D/probe.png', dpi=300)

# Construcción de la cuadrícula y las mediciones de superficie simuladas según la Sección 7.4.
Z = np.zeros(W.shape)
ZP_col = np.zeros((Nz, 2**(nextpow2(2*Nx) + 1) - 2*Nx))
ZP_row = np.zeros((2**(nextpow2(2*Nz) + 1) - 2*Nz, 2**(nextpow2(2*Nx) + 1)))


## Matriz P 
P = np.block([
        [Z, W, ZP_col],
        [Z, Z, ZP_col],
        [ZP_row]
    ])

Ly, Lx = P.shape

# Crear vectores kx y ky
kx = (np.arange(Lx) / Lx) * np.pi
ky = (np.arange(Ly) / Ly) * np.pi

# Crear la matriz de frecuencia
f = np.zeros((Lx, Ly))
for kxi in range(Lx):
    for kyi in range(Ly):
        f[kxi, kyi] = np.sqrt(kx[kxi]**2 + ky[kyi]**2)

# Simular ondas acústicas ideales sin atenuación.
Pdet = np.zeros((P.shape[0], len(f)))
P_hat = dct(dct(P.T, norm='ortho').T, norm='ortho')  # Transformada coseno 2D
for t in range(P.shape[0]):
    cost = np.cos(c0 * f * t)
    Pcos = P_hat * cost.T  # Construir la presión acústica evaluada en la superficie
    Pt = idct(idct(Pcos.T, norm='ortho').T, norm='ortho') / 3
    Pdet[t, :] = Pt[0, :]

# Simulación de ondas acústicas con atenuación
MODEL = SimpleNamespace(
    dt=1,             # Ancho de cada paso de tiempo
    dz=1,             # Ancho del paso espacial
    Nz=P.shape[0],    # Área de simulación, tamaño de las matrices D, M4 y M5
    idx_l=0,          # Borde izquierdo del área física, ubicación del sensor
    idx_r=P.shape[0], # Borde derecho del área física
    Nd=P.shape[0],    # Longitud del vector d
    Ny=P.shape[0],    # Longitud del vector de medición
    sigma_q2=1e-11,   # Varianza del ruido en la ecuación de transición de estado
    sigma_w2=1e-11,   # Varianza del ruido de medición
    regularization_term=0  # Término de regularización para el estimador LS (no relevante aquí)
)
MODEL.idx_l += 1  # Transformar el índice del punto de la cuadrícula en un índice de los vectores
MODEL.idx_r += 1  # Transformar el índice del punto de la cuadrícula en un índice de los vectores

PROBE = SimpleNamespace(
    tau=77e-12,      # Tiempo de relajación del medio
    chi=3e-2,
    c0=c0,           # Velocidad de propagación en el medio
    beta=1,
    Cp=1,
    mu=np.zeros(MODEL.Nd)  # No es importante aquí, solo queremos calcular la matriz H, en la que mu no está involucrado
)

signal_Nr = 5  # 2 ... chirp, 5 ... Gaussian
SIGNAL = SimpleNamespace(
    Ni=60,               # Longitud de la señal de entrada
    Omega_min=0.1*np.pi, # Frecuencia más baja de un chirp
    Omega_max=0.25*np.pi # Frecuencia más alta de un chirp
)
SIGNAL.i = generate_standard_input_signal(signal_Nr, SIGNAL, MODEL)
MODEL = generate_ssm_model(MODEL, PROBE)
MODEL = generate_linear_model(MODEL, SIGNAL, PROBE)

P_surf = np.zeros_like(Pdet)
P_surf_without_noise = np.zeros_like(Pdet)

for i in range(P_surf.shape[1]):
    print(f'Iteration: {i + 1} / {P_surf.shape[1]}')
    MODEL.d = Pdet[:, i]
    PROBE.mu = mu_from_d(MODEL, MODEL.d)
    MODEL = generate_ssm_model(MODEL, PROBE)
    SIGNAL = generate_measurements(SIGNAL, MODEL)
    P_surf[:, i] = SIGNAL.y.flatten()
    P_surf_without_noise[:, i] = SIGNAL.y_without_meas_and_proc_noise.flatten()

P_true = P
SNR = 10 * np.log10(np.sum(P_surf_without_noise**2) / np.sum((P_surf - P_surf_without_noise)**2))

# Mostrar la temperatura de la superficie de la sonda
plt.figure(1)
plt.imshow(P_surf.T, aspect='auto')
plt.xlabel('t', fontsize=fsize)
plt.ylabel('y', fontsize=fsize)
plt.xlim([0, 500])
plt.ylim([0, 500])
plt.gca().tick_params(axis='both', which='major', labelsize=fsize)
plt.title('Measured surface temperature', fontsize=12)
plt.colorbar()
plt.savefig('img/2D/surface_temperature.png', dpi=300)

# Guardar los datos
data = {
    'P_surf': P_surf,
    'P_surf_without_noise': P_surf_without_noise,
    'Pdet': Pdet,
    'px': px,
    'py': py,
    'P_true': P_true,
    'Nx': Nx,
    'Nz': Nz,
    'MODEL': MODEL,
    'PROBE': PROBE,
    'SIGNAL': SIGNAL,
    'SNR': SNR
}

if signal_Nr == 5:
    with open('2D/Data/pressure_dist_pulse.pkl', 'wb') as f:
        pickle.dump(data, f)
elif signal_Nr == 2:
    with open('./Data/pressure_dist_chirp.pkl', 'wb') as f:
        pickle.dump(data, f)