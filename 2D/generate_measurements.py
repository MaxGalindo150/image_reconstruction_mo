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
Nz = 100  # Depth resolution
Nx = 100  # Spatial resolution along the scanning direction
Nt = 100
c0 = 1
sigma = 1
fsize = 12  # Font size for figure labels

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
rr, cc = polygon([py[2], py[2], py[2] + s, py[2] + s], [px[2], px[2] + s, px[2] + s, px[2]])
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


# Constructiong the grid and the simulated surface measurements according to Section 7.4.

Z = np.zeros(W.shape)
ZP_col = np.zeros((Nz, 2**(nextpow2(2*Nx) + 1) - 2*Nx))
ZP_row = np.zeros((2**(nextpow2(2*Nz) + 1) - 2*Nz, 2**(nextpow2(2*Nx) + 1)))

P = np.block([
        [Z, W, ZP_col],
        [Z, Z, ZP_col],
        [ZP_row]
    ])


Ly, Lx = P.shape
print(f'P shape: {P.shape}')

# Create kx and ky vectors
kx = (np.arange(Lx) / Lx) * np.pi
ky = (np.arange(Ly) / Ly) * np.pi

# Create the frequency matrix
f = np.zeros((Lx, Ly))
for kxi in range(Lx):
    for kyi in range(Ly):
        f[kxi, kyi] = np.sqrt(kx[kxi]**2 + ky[kyi]**2)

# Simulate ideal acoustic waves without attenuation.
Pdet = np.zeros((P.shape[0], len(f)))
P_hat = dct(dct(P.T, norm='ortho').T, norm='ortho')  # 2D cosine transform
for t in range(P.shape[0]):
    cost = np.cos(c0 * f * t)
    Pcos = P_hat * cost.T  # Construir la presión acústica evaluada en la superficie
    Pt = idct(idct(Pcos.T, norm='ortho').T, norm='ortho') / 3
    Pdet[t, :] = Pt[0, :]

# Simulating acoustic waves with attenuation
MODEL = SimpleNamespace(
    dt=1,             # Width of each time step
    dz=1,             # Width of space step
    Nz=P.shape[0],    # Sim Area, Size of matrices D, M4, and M5
    idx_l=0,          # Left border of physical area, location of the sensor
    idx_r=P.shape[0], # Right border of physical area
    Nd=P.shape[0],    # Length of the vector d
    Ny=P.shape[0],    # Length of the measurement vector
    sigma_q2=1e-11,   # Variance of the noise in the state transition equation
    sigma_w2=1e-11,   # Variance of the measurement noise
    regularization_term=0  # Scalar used for regularization of the LS estimator (not relevant here)
)
MODEL.idx_l += 1  # Transforming the grid point index into an index of the vectors
MODEL.idx_r += 1  # Transforming the grid point index into an index of the vectors


PROBE = SimpleNamespace(
    tau=77e-12,      # Relaxation time of medium
    chi=3e-2,
    c0=c0,           # Propagation speed in medium
    beta=1,
    Cp=1,
    mu=np.zeros(MODEL.Nd)  # Not important here, we just want to compute the matrix H, in which mu is not involved
)

signal_Nr = 5  # 2 ... chirp, 5 ... Gaussian
SIGNAL = SimpleNamespace(
    Ni=60,               # Length of the input signal
    Omega_min=0.1*np.pi, # Lowest frequency of a chirp
    Omega_max=0.25*np.pi # Highest frequency of a chirp
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

# Displaying the surface temperature of the probe
plt.figure(1)
plt.imshow(P_surf.T, aspect='auto')
plt.xlabel('t', fontsize=fsize)
plt.ylabel('y', fontsize=fsize)
plt.gca().tick_params(axis='both', which='major', labelsize=fsize)
plt.title('Measured surface temperature', fontsize=12)
plt.colorbar()
plt.savefig('img/2D/surface_temperature.png', dpi=300)

print(f'P_surf: {P_surf}')

# Save the data
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
    with open('./Data/pressure_dist_pulse.pkl', 'wb') as f:
        pickle.dump(data, f)
elif signal_Nr == 2:
    with open('./Data/pressure_dist_chirp.pkl', 'wb') as f:
        pickle.dump(data, f)
