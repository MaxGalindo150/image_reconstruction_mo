from SSM.Set_Settings import SimulationSettings
import matplotlib.pyplot as plt

simulation_settings = SimulationSettings(1)

MODEL, PROBE, SIGNAL = simulation_settings.get_settings()

plt.figure()
plt.plot(PROBE.mu, 'b-', linewidth=4, label='True µ')
plt.legend()
plt.savefig('output_plot.png')