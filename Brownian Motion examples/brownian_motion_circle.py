import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from processes import WienerProcess

# See Hsu example
SEED = 123456
wiener_process = WienerProcess()
dt = 0.01
tf = 6
n = int(tf / dt)
t = np.linspace(0, tf, n)
theta = wiener_process.simulate(dt, tf)

x = np.cos(theta)
y = np.sin(theta)

# Animate motion
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)), color='k', alpha=0.2)
line, = ax.plot(x[0], y[0], color='red', alpha=0.4)
particle =  ax.scatter(x[0], y[0], color='red')
ax.set_title(f'Brownian Motion on a Circle t={round(t[0], 3)}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

def animate(i):
    line.set_xdata(x[:i])
    line.set_ydata(y[:i])
    particle.set_offsets([x[i], y[i]])
    ax.set_title(f'Brownian Motion on a Circle t={round(t[i], 3)}')


anim_gif = animation.FuncAnimation(fig, animate, interval=dt * 1000 * 8, frames=np.arange(1, n))
anim_gif.save('brownian_motion_on_circle.gif')
