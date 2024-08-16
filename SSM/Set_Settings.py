from types import SimpleNamespace
from SSM.Generate_Standard_Absorption_Profiles import generate_standard_absorption_profiles
from SSM.Generate_Standard_Input_Signal import generate_standard_input_signal

class SimulationSettings:
    def __init__(self, example):
        self.model = SimpleNamespace()
        self.probe = SimpleNamespace()
        self.signal = SimpleNamespace()
        self.set_settings(example)

    def set_settings(self, example):
        if example == 1:
            self.model = SimpleNamespace(**{
                'dt': 1e-9,
                'dz': 30e-7,
                'Nz': 20,
                'idx_l': 1,
                'idx_r': 21,
                'Nd': 20,
                'Ny': 100,
                'sigma_q2': 1e-28,
                'sigma_w2': 1e-10,
                'regularization_term': 0
            })
            profile_nr = 2
            self.probe = SimpleNamespace(**{
                'tau': 77e-12,
                'chi': 3e-2,
                'c0': 1500,
                'beta': 1,
                'Cp': 1,
                'mu': generate_standard_absorption_profiles(profile_nr, self.model)
            })
            signal_nr = 5
            self.signal = SimpleNamespace(**{
                'Ni': 50,
                'Omega_min': 0.1 * 3.141592653589793,
                'Omega_max': 0.25 * 3.141592653589793
            })
            self.signal.i = generate_standard_input_signal(signal_nr, self.signal, self.model)
        elif example == 2:
            self.model = SimpleNamespace(**{
                'dt': 1e-9,
                'dz': 2e-5,
                'Nz': 200,
                'idx_l': 1,
                'idx_r': 201,
                'Nd': 200,
                'Ny': 2000,
                'sigma_q2': 1e-28,
                'sigma_w2': 1e-10,
                'regularization_term': 1.5e-11
            })
            profile_nr = 3
            self.probe = SimpleNamespace(**{
                'tau': 77e-12,
                'chi': 3e-2,
                'c0': 1500,
                'beta': 1,
                'Cp': 1,
                'mu': generate_standard_absorption_profiles(profile_nr, self.model)
            })
            signal_nr = 5
            self.signal = SimpleNamespace(**{
                'Ni': 20,
                'Omega_min': 0.7 * 3.141592653589793,
                'Omega_max': 0.99 * 3.141592653589793
            })
            self.signal.i = generate_standard_input_signal(signal_nr, self.signal, self.model)
        # Add other cases similarly
        else:
            self.model = SimpleNamespace()
            self.probe = SimpleNamespace()
            self.signal = SimpleNamespace()

    def get_settings(self):
        return self.model, self.probe, self.signal