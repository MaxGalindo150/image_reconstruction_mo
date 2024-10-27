import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

def fsaft_operator(Ps, dt, dx, c0):
    """
    Implementa el algoritmo SAFT en el dominio de la frecuencia.

    Parámetros de entrada:
    Ps : Imagen de ultrasonido de superficie sobre la cual se aplica el operador FSAFT.
    dt : Periodo de muestreo temporal.
    dx : Intervalo de muestreo espacial a lo largo de la dirección de escaneo.
    c0 : Velocidad del sonido en la sonda.

    Parámetros de salida:
    P0_est : Imagen reconstruida usando el algoritmo FSAFT.
    """
    nt, ny = Ps.shape
    

    # FFT
    fftRF = np.fft.fftshift(np.fft.fft2(Ps, s=(nt, ny))) / (nt * ny)
    
    plt.figure()
    plt.imshow(np.real(fftRF), aspect='auto', cmap='gray')
    plt.colorbar()
    plt.title('FFT')
    plt.savefig('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/img/2D/fftRF.png', dpi=500)

    

    # Interpolación lineal
    fs = 1 / dt  # Frecuencia de muestreo de las señales de ultrasonido
    f = (np.arange(-nt/2, nt/2) * fs / nt)
    kx = np.arange(-ny/2, ny/2) / dx / ny
    kx, f = np.meshgrid(kx, f)
    fkz = c0 * np.sign(f) * np.sqrt(kx**2 + (f / c0)**2)
    nearestind, colind = nearest(f, fkz)
    INT = csr_matrix((np.ones(len(nearestind)), (colind, nearestind)), shape=(ny * nt, ny * nt))
    fftRF2 = INT.dot(fftRF.ravel()).reshape(nt, ny)
    
    plt.figure()
    plt.imshow(np.real(fftRF2), aspect='auto', cmap='gray')
    plt.xlabel('scanning direction x', fontsize=12)
    plt.ylabel('time t', fontsize=12)
    plt.grid(True)
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalente a 'axis xy' en MATLAB
    plt.title('fft', fontsize=12)
    plt.colorbar()
    plt.savefig('/home/mgalindo/max/maestria/tesis/image_reconstruction_mo/img/2D/fftRF2.png', dpi=500)
    
    

    # IFFT & Migrated Ps
    P0_est = np.fft.ifft2(np.fft.ifftshift(fftRF2))
    

    return P0_est

def nearest(f, fkz):
    """
    Encuentra los índices más cercanos para la interpolación lineal.
    """
    nf, mf = f.shape
    l = int(np.log2(nf))
    nearestind = []
    colind = []

    for k in range(mf):
        for i in range(nf):
            if fkz[i, k] >= f[0, k] and fkz[i, k] <= f[-1, k]:
                ind = 0
                for j in range(1, l + 1):
                    if fkz[i, k] >= f[min(ind + 2**(l - j), nf - 1), k]:
                        ind += 2**(l - j)
                # Asegurarse de que `ind + 1` no exceda el tamaño de `f`
                if ind + 1 < nf and abs(fkz[i, k] - f[ind, k]) < abs(fkz[i, k] - f[ind + 1, k]):
                    nearestind.append((k * nf) + ind)
                    colind.append((k * nf) + i)
                else:
                    nearestind.append((k * nf) + min(ind + 1, nf - 1))
                    colind.append((k * nf) + i)

    return np.array(nearestind), np.array(colind)

