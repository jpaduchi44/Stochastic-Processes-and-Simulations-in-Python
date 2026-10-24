import numpy as np
import matplotlib.pyplot as plt

class DiscreteMarkovChain:
  def __init__(self, initial_prob, transition_prob):
    self.n_states = initial_prob.size
    self.initial_prob = initial_prob
    self.transition_prob = transition_prob

  def simulate(self, steps):
    chain = np.repeat(-1, steps)
    chain[0] = np.random.choice(self.n_states, p=self.initial_prob)
    for i in range(steps - 1):
      chain[i + 1] = np.random.choice(self.n_states, p=self.transition_prob[chain[i], :])
    return chain

class WienerProcess:
  def __init__(self):
    pass

  def simulate(self, dt, t_fin):
    steps = int(t_fin / dt)
    dw = np.random.normal(loc=0, scale=np.sqrt(dt), size=steps-1)
    w = np.hstack((0, np.cumsum(dw)))
    return w

class PoissonProcess:
  def __init__(self, rate):
    self.rate = rate

  def simulate(self, t_fin):
    t = np.array([0])
    while True:
      increment = np.random.exponential(scale=self.rate)
      val = t[-1] + increment
      if val > t_fin:
        t = np.append(t, t_fin)
        break
      else:
        t = np.append(t, val)
    return t, np.append(np.arange(t.size-1), t.size-2)

  def plot(self, t, y, color='b', alpha=1):
    for i in range(t.size - 1):
      plt.plot([t[i], t[i+1]], [y[i], y[i]], color=color, alpha=alpha)

def euler_maruyama(x_init, t_fin, mu, sigma, n):
  dt = t_fin / n
  t = np.linspace(0, t_fin, n)
  x = np.zeros(n)
  x[0] = x_init
  for i in range(n - 1):
    dw = np.sqrt(dt) * np.random.normal(0, 1, 1)
    x[i + 1] = x[i] + mu(x[i], t[i]) * dt + sigma(x[i], t[i]) * dw
  return t, x

def euler_maruyama_d(x_init, t_fin, mu, sigma, n):
  d = x_init.shape[0]
  dt = t_fin / n
  t = np.linspace(0, t_fin, n)
  x = np.zeros((d, n))
  x[:, 0] = x_init
  for i in range(n - 1):
    dw = np.sqrt(dt) * np.random.normal(0, 1, d)
    x[:, i + 1] = x[:, i] + mu(x[:, i], t[i]) * dt + sigma(x[:, i], t[i]) @ dw
  return t, x

