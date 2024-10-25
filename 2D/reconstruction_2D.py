import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from External_lib.Tikhonov import tikhonov
from External_lib.L_Curve import l_curve
from Migration.fsaft_operator import fsaft_operator
from MO.NSGA_II_estimation import nsga2_estimation

# Load data from 2D/Data/pressure_dist_pulse.pkl
with open('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/2D/Data/pressure_dist_pulse.pkl', 'rb') as f:
    data = pickle.load(f)

c0 = 1
Nzz, Nxx = data['P_surf'].shape
Nt = Nzz
tt = np.arange(Nt)
dx = 1
dt = 1


plt.figure(1)
plt.imshow(data['Pdet'][0:Nzz, :Nxx], aspect='auto', cmap='gray')
plt.xlabel('scanning direction x', fontsize=12)
plt.ylabel('time t', fontsize=12)
plt.grid(True)
plt.axis('image')
plt.xlim(0, 200)
plt.ylim(0, 200)
#plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.title('Surface measurements without attenuation', fontsize=12)
plt.colorbar()
plt.savefig('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/img/2D/surface_measurements_wo_attenuation.png', dpi=300)

# The surface measurements
plt.figure(2)
plt.imshow(data['P_surf'][0:Nzz, :Nxx], aspect='auto', cmap='gray')
plt.xlabel('scanning direction x', fontsize=12)
plt.ylabel('time t', fontsize=12)
plt.grid(True)
plt.axis('image')
plt.xlim(0, 200)
plt.ylim(0, 200)
#plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.title('Surface measurements with attenuation', fontsize=12)
plt.colorbar()
plt.savefig('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/img/2D/surface_measurements_wo_attenuation.png', dpi=300)


# ---- Regularized inversion ----
print(1/(20*data['SNR']))
P_virt = np.zeros(data['P_surf'].shape)
U, s, V = np.linalg.svd(data['MODEL'].H, full_matrices=False)
L = np.eye(data['MODEL'].H.shape[1])
s = s.reshape(-1, 1)
for i in range(data['P_surf'].shape[1]):
    print(f'Iteration: {i + 1} / {data["P_surf"].shape[1]}')
    b = data['P_surf'][:, i]
    #rho, eta, reg_corner_tikh, rho_c, eta_c, reg_param = l_curve(U, s, b, "Tikh", L, V)
    #d_est_tikh = tikhonov(data['MODEL'].H, b, 1/(20*data['SNR']), dim=2)
    #P_virt[:, i] = d_est_tikh.flatten()
    d_est_nsga2 = nsga2_estimation(data['MODEL'], data['PROBE'], data['SIGNAL'], n_var=512, b=b.reshape(512,1), lambda_tikhonov=1/(20*data['SNR']))
    print("NSGA-II d: ", d_est_nsga2.flatten())
    #print("Tikhonov regularization d: ", tikhonov_aprox.flatten())
    P_virt[:, i] = d_est_nsga2.flatten()
    
plt.figure(2)
plt.imshow(P_virt, aspect='auto', cmap='gray')
plt.xlabel('scanning direction x', fontsize=12)
plt.ylabel('time t', fontsize=12)
plt.grid(True)
plt.axis('image')
plt.xlim(0, 500)
plt.ylim(0, 500)
#plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.title('Surface measurements with attenuation', fontsize=12)
plt.colorbar()
plt.savefig('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/img/2D/P_virt.png', dpi=300)


# Applying the F-SAFT algorithm for recosntruction the pressure distribution the material
P_rec_fsaft = fsaft_operator(P_virt, dt, dx, c0)

#Contar el número de ceros
num_zeros = np.count_nonzero(P_rec_fsaft == 0)

# Calcular el porcentaje de ceros
total_elements = P_rec_fsaft.size
percentage_zeros = (num_zeros / total_elements) * 100

print(f"El porcentaje de ceros en la matriz P_rec_fsaft es: {percentage_zeros:.2f}%")


plt.figure(5)
plt.imshow(np.real(P_rec_fsaft), aspect='auto', cmap='gray', origin='lower')
plt.xlabel('scanning direction x', fontsize=12)
plt.ylabel('depth z', fontsize=12)
plt.colorbar()
plt.grid(True)
plt.title('FSAFT-reconstruction without attenuation', fontsize=12)
plt.savefig('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/img/2D/antes.png', dpi=300)

P_rec_fsaft = np.maximum(P_rec_fsaft[:data['Nz'], data['Nx']:2*data['Nx']], 0).T

# Contar el número de ceros
num_zeros = np.count_nonzero(P_rec_fsaft == 0)

# Calcular el porcentaje de ceros
total_elements = P_rec_fsaft.size
percentage_zeros = (num_zeros / total_elements) * 100

print(f"El porcentaje de ceros en la matriz P_rec_fsaft es: {percentage_zeros:.2f}%")

plt.figure(5)
plt.imshow(np.real(P_rec_fsaft), aspect='auto', cmap='gray', origin='lower')
plt.xlabel('scanning direction x', fontsize=12)
plt.ylabel('depth z', fontsize=12)
plt.colorbar()
plt.grid(True)
plt.title('FSAFT-reconstruction without attenuation', fontsize=12)
plt.savefig('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/img/2D/sFSAFT-reconstruction-without-attenuation.png', dpi=300)

