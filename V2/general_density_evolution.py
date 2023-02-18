import numpy as np
import matplotlib.pyplot as plt
from strings_FLRW import *

def compute_general_density_evolution(w, rho_crit, omega_m0, omega_phantom0):
    sigma = np.linspace(0, np.pi, 2000)
    flrw = FLRW_general(w, 3, rho_crit, omega_phantom0)
    string = String(sigma, flrw)

    t0 = 0
    a0 = 1
    rho_B0 = omega_m0*rho_crit

    x0 = np.cos(sigma)/a0
    y0 = np.zeros_like(sigma)
    xdot0 = np.zeros_like(sigma)
    ydot0 = np.cos(sigma)/a0
    epsilon0 = np.ones_like(sigma)/a0
    rho_B0 = np.ones_like(sigma)*rho_B0
    a0 = np.ones_like(sigma)

    u0 = np.array([x0, y0, xdot0, ydot0, epsilon0, rho_B0, a0])

    t_unbound, u = string.solve(u0, delta0 = t0, type = "general")

    print(t_unbound)

    t     = u[0, :]
    E     = u[1, :]
    rho_B = u[2, :]
    a     = u[3, :]

    fig, ax = plt.subplots()
    ax.plot(t, E, "m-")
    ax.plot([t[0], t[-1]], [2*E[0], 2*E[0]], "k--")
    ax.set_xlabel(r"$t'$")
    ax.set_ylabel(r"$E/T_0$")

    fig, ax = plt.subplots()
    ax.plot(t, a, "c-", label = r"$a(t')$")
    ax.set_xlabel(r"$t'$")
    ax.set_ylabel(r"$a(t')$")
    ax.set_title(r"$\Omega_{m0} = %s, w = %s$" % (omega_m0, w))

    a_EdS = a[0]*(1 + 3/2*flrw.H0*(np.array(t) - t[0]))**(2/3)

    ax.plot(t, a_EdS, "k--", label = r"$a_\mathrm{EdS} \propto \left(t'\right)^{2/3}$")
    ax.legend()
    fig.savefig("images/a(t)_w_2_omegam0_099.pdf", bbox_inches = "tight")

    fig, ax = plt.subplots()
    ax.plot(t, rho_B/rho_B[0], "r-", label = r"$\rho_B(t')$")
    ax.set_xlabel(r"$t'$")
    ax.set_ylabel(r"$\rho_B/\rho_{B0}$")
    ax.set_title(r"$\Omega_{m0} = $ %s, $w = $ %s" % (omega_m0, w))

    ax.plot(t, np.array(a)**(-3), "k--", label = r"$\rho_B \propto a^{-3}$")
    ax.legend()
    fig.savefig("images/density_general_w_2_omegam0_099.pdf", bbox_inches = "tight")
    plt.show()

omega_m0 = 0.99

compute_general_density_evolution(-2, 1, omega_m0, 1 - omega_m0)
