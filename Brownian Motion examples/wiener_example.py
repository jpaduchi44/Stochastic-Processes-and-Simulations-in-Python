import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import animation
from processes import WienerProcess

# Simulate m samples of the standard Brownian Motion
SEED = 123456

wiener_process = WienerProcess()
n = 150
t_fin = 3
dt = t_fin / n
t = np.linspace(0, t_fin, n)

m = 15
samples = np.zeros((m, n))

np.random.seed(SEED)
for i in range(m):
  samples[i, :] = wiener_process.simulate(dt, t_fin)

# Compute pdf at each time step
k = 80
x_lim = 5
x = np.linspace(-x_lim, x_lim, k)
pdf_values = np.zeros((k, n))
for i in range(1, n):
  pdf_values[:, i] = norm.pdf(x, 0, t[i])

t_mesh, x_mesh = np.meshgrid(t, x)

# Plot paths
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(m):
  ax.plot(t, samples[i, :])
ax.set_title('Brownian Motion sample paths')
ax.set(xlim=(0, t_fin), xlabel='t')
ax.set(ylim=(np.min(samples) - 0.2, np.max(samples) + 0.2), ylabel='W')
fig.savefig('brownian_paths.jpeg')

# Animate pdf and samples
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

sample_paths = [ax.plot(t[0], samples[i, 0], [0])[0] for i in range(m)]
pdf_surface =  ax.plot_surface(np.reshape(t_mesh[:, 0], (k, 1)), 
                               np.reshape(x_mesh[:, 0], (k, 1)), 
                               np.reshape(pdf_values[:, 0], (k, 1)), 
                               color='lightcyan', alpha=0.4)
ax.set_title(f'Brownian Motion Samples and Distribution\n t={round(t[0], 3)}')
ax.set(xlim3d=(0, t_fin), xlabel='t')
ax.set(ylim3d=(-x_lim, x_lim), ylabel='W')
ax.set(zlim3d=(0, 3), zlabel='pdf')

def animate(i):
  global pdf_surface
  for j, path in enumerate(sample_paths):
    path.set_xdata(t[:i])
    path.set_ydata(samples[j, :i])
    path.set_3d_properties(np.zeros(i))
  pdf_surface.remove()
  pdf_surface = ax.plot_surface(t_mesh[:, :i], x_mesh[:, :i], pdf_values[:, :i], color='lightcyan', alpha=0.4)
  ax.set_title(f'Brownian Motion Samples and Distribution\n t={round(t[i], 3)}')

anim_gif = animation.FuncAnimation(fig, animate, interval=dt * 1000 * 4, frames=np.arange(1, n))
anim_gif.save('brownian_paths_and_distribution.gif')

# Plot paths and density
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

for i in range(m):
  ax.plot(t, samples[i, :], np.zeros(n))
pdf_surface =  ax.plot_surface(t_mesh, 
                               x_mesh, 
                               pdf_values, 
                               color='lightcyan', alpha=0.4)
ax.set_title(f'Brownian Motion Samples and Distribution')
ax.set(xlim3d=(0, t_fin), xlabel='t')
ax.set(ylim3d=(-x_lim, x_lim), ylabel='W')
ax.set(zlim3d=(0, 3), zlabel='pdf')
fig.savefig('brownian_paths_and_distribution.jpeg')
