import matplotlib.pyplot as plt
from SSM.Set_Settings import set_settings

MODEL, PROBE, SIGNAL = set_settings(example=1)
plt.figure()
plt.plot(PROBE.mu, 'b-', linewidth=4, label='True µ')
plt.legend()
plt.savefig('output_plot.png')