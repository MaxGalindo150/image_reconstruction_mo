import numpy as np
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift
from scipy.sparse import csr_matrix

def fsaft_operator(Ps, dt, dx, c0):
    """
    Compute the FSaft operator.
    
    Parameters
    ----------
    Ps : numpy.ndarray
        The pressure measurements.
    dt : float
        The time step.
    dx : float
        The spatial step.
    c0 : float
        The speed of sound.
    
    Returns
    -------
    numpy.ndarray
        The FSaft operator.
    """

    # Exploding Reflector Model velocity 
    nt, ny = Ps.shape
    global INT
    
    # FFT
    # fftRF = fftshift(fft2(Ps, (nt, ny))) / nt / ny
    fftRF = fftshift(fft2(Ps, (nt, ny)))

    # Linear interpolation   
    if 'INT' not in globals() or INT is None:
        fs = 1 / dt  # Sampling frequency of the ultrasound signals.
        #f = (np.arange(-nt/2, nt/2) * fs / nt)
        f = np.fft.fftfreq(nt, d=dt)
        kx = np.fft.fftfreq(ny, d=dx)
        #kx = np.arange(-ny/2, ny/2) / dx / ny
        #print(f'f shepe: {f.shape}, kx shape: {kx.shape}')
        kx, f = np.meshgrid(kx, f)
        fkz = c0 * np.sign(f) * np.sqrt(kx**2 + f**2 / c0**2)
        nearestind, colind = nearest(f, fkz)
        INT = csr_matrix((np.ones(len(nearestind)), (colind, nearestind)), shape=(ny * nt, ny * nt))
    
    fftRF2 = INT @ fftRF.flatten()
    fftRF2 = fftRF2.reshape(nt, ny)
    
    # IFFT & Migrated Ps
    P0_est = ifft2(ifftshift(fftRF2))
    return P0_est

def nearest(f, fkz):
    nf, mf = f.shape
    l = int(np.log2(nf))
    nearestind = np.zeros(nf * mf, dtype=int)
    colind = np.zeros(nf * mf, dtype=int)
    top = 0

    for k in range(mf):
        for i in range(nf):
            ind = 0
            if fkz[i, k] >= f[0, k] and f[-1, k] >= fkz[i, k]:
                for j in range(l):
                    if fkz[i, k] >= f[ind + 2**(l - j - 1), k]:
                        ind += 2**(l - j - 1)
                if abs(fkz[i, k] - f[ind, k]) < abs(fkz[i, k] - f[ind + 1, k]):
                    nearestind[top] = k * nf + ind
                    colind[top] = k * nf + i
                else:
                    nearestind[top] = k * nf + ind + 1
                    colind[top] = k * nf + i
                top += 1

    nearestind = nearestind[:top]
    colind = colind[:top]
    return nearestind, colind
