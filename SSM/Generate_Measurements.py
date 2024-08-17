import numpy as np
from types import SimpleNamespace

def generate_measurements(signal: SimpleNamespace, model: SimpleNamespace, probe: SimpleNamespace) -> SimpleNamespace: #sinal
    y = np.zeros((model.Ny, 1))
    y_without_meas_noise = np.zeros((model.Ny, 1))
    x = np.zeros((2*model.Nz, 1))
    x_without_proc_noise = np.zeros((2*model.Nz, 1))
    y_without_meas_and_proc_noise = np.zeros((model.Ny, 1))
    
    signal.u = signal.i.flatten() - np.concatenate(([0], signal.i[:-1].flatten()))
    input_signal = np.concatenate((signal.u[:, np.newaxis], np.zeros((signal.Ny - signal.Ni, 1))))
    for iter in range(signal.Ny - 1):
        x = model.A @ x + model.g * input_signal[iter] + np.sqrt(model.sigma_q2) * np.concatenate((np.random.randn(model.Nz, 1), np.zeros((model.Nz, 1))))
        x_without_proc_noise = model.A @ x_without_proc_noise + model.g * input_signal[iter]
        y[iter+1] = model.c.T @ x + np.sqrt(model.sigma_w2) * np.random.randn(1,1)
        y_without_meas_noise[iter+1] = model.c.T @ x
        y_without_meas_and_proc_noise[iter+1] = model.c.T @ x_without_proc_noise
    
    signal.y = y
    signal.y_without_meas_noise = y_without_meas_noise
    signal.y_without_meas_and_proc_noise = y_without_meas_and_proc_noise

    return signal
