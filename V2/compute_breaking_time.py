import numpy as np
import matplotlib.pyplot as plt
from strings_FLRW import *

H0 = 3.77e-42

def compute_breaking_time(delta0_fraction, w):
    sigma = np.linspace(0, np.pi, 2000)
    flrw = FLRW_Phantom(w = w, H0 = H0)

    delta0 = delta0_fraction/(-flrw.alpha)
    a0 = flrw.a(delta0)
    #print(a0)

    x0       = np.cos(sigma)/a0
    y0       = np.zeros_like(sigma)
    xdot0    = np.zeros_like(sigma)
    ydot0    = np.cos(sigma)/a0
    epsilon0 = np.ones_like(sigma)/a0
    rho_B0   = np.ones_like(sigma)

    u0 = np.array([x0, y0, xdot0, ydot0, epsilon0, rho_B0])

    string = String(sigma, flrw)
    delta_unbound, u = string.solve(u0, delta0 = delta0)

    print(delta_unbound)

    fig, ax = plt.subplots()
    delta = u[0, :]
    E = u[1, :]
    ax.plot(delta, E, "m-")
    ax.plot([delta[0], delta[-1]], [2*E[0], 2*E[0]], "k--")
    plt.show()

compute_breaking_time(0.35, -1.03)
