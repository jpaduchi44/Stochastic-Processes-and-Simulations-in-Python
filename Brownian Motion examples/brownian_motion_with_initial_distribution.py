import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import animation
from processes import WienerProcess

# Simulate m samples of the standard Brownian Motion
# for x and y components
SEED = 1

wiener_process = WienerProcess()
n = 150
t_fin = 0.3
dt = t_fin / n
t = np.linspace(0, t_fin, n)

m = 20
x_brown = np.zeros((m, n))
y_brown = np.zeros((m, n))

np.random.seed(SEED)
for i in range(m):
  x_brown[i, :] = wiener_process.simulate(dt, t_fin)
  y_brown[i, :] = wiener_process.simulate(dt, t_fin)

# Combine with initial distribution
initial_states = [np.array([1, 0]),
                  np.array([-1, 1]),
                  np.array([-1, -1])]
probabilities = np.array([0.4, 0.4, 0.2])
for i in range(m):
  choice = np.random.choice(np.arange(len(initial_states)), p=probabilities)
  x_brown[i, :] += initial_states[choice][0]
  y_brown[i, :] += initial_states[choice][1]

# Compute pdf at each time step
k = 100
x_lim = 3
y_lim = 3
x, y = np.meshgrid(np.linspace(-x_lim, x_lim, k), np.linspace(-y_lim, y_lim, k))
xy = np.dstack((x, y))
pdf_values = np.zeros((k, k, n))
for i in range(1, n):
  for j, state in enumerate(initial_states):
    pdf_values[:, :, i] += probabilities[j] * multivariate_normal.pdf(xy, mean=state, cov=[[t[i], 0], [0, t[i]]])

# Animate pdf and samples
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

sample_paths = [ax.plot(x_brown[i, 0], y_brown[i, 0], [0])[0] for i in range(m)]
pdf_surface =  ax.plot_surface(x, y, pdf_values[:, :, 0], color='lightcyan', alpha=0.5)

ax.set_title(f'2d Brownian Motion with initial distribution\n t={round(t[0], 3)}')
ax.set(xlim3d=(-x_lim, x_lim), xlabel='x')
ax.set(ylim3d=(-y_lim, y_lim), ylabel='y')
ax.set(zlim3d=(0, 3), zlabel='pdf')

def animate(i):
  global pdf_surface
  for j, path in enumerate(sample_paths):
    path.set_xdata(x_brown[j, :i])
    path.set_ydata(y_brown[j, :i])
    path.set_3d_properties(np.zeros(i))
  pdf_surface.remove()
  pdf_surface = ax.plot_surface(x, y, pdf_values[:, :, i], color='lightcyan', alpha=0.5)
  ax.set_title(f'2d Brownian Motion with initial distribution\n t={round(t[i], 3)}')

anim_gif = animation.FuncAnimation(fig, animate, interval=dt * 1000 * 20, frames=np.arange(1, n))
anim_gif.save('with_initial_distribution.gif')
