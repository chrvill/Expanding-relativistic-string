import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from strings_FLRW import *

class DensityEvolution:
    def __init__(self, w, flrw, unbinding_time_interpolator):
        self.w = w
        self.unbinding_time_interpolator = unbinding_time_interpolator
        #self.flrw = FLRW_Phantom(w = w, H0 = 1)
        self.flrw = flrw

    def rhs(self, rho_B, delta):
        tau = self.unbinding_time_interpolator(delta*(-self.flrw.alpha), self.w)[0]
        return -3*self.flrw.H(delta)*rho_B + rho_B/tau

    def RK4(self, rho_B, delta, ddelta):
        k1 = self.rhs(rho_B, delta)
        k2 = self.rhs(rho_B + k1*ddelta/2, delta + ddelta/2)
        k3 = self.rhs(rho_B + k2*ddelta/2, delta + ddelta/2)
        k4 = self.rhs(rho_B + k3*ddelta, delta + ddelta)

        return 1/6*(k1 + 2*(k2 + k3) + k4)

    def solve(self, rho_B0, delta):
        n = len(delta)
        ddelta = (delta[-1] - delta[0])/n

        rho_B = np.zeros_like(delta)
        rho_B[0] = rho_B0

        for i in range(n - 1):
            rho_B[i + 1] = rho_B[i] + self.RK4(rho_B[i], delta[i], ddelta)*ddelta

        return rho_B

def readData(data_filename):
    """
    Loads all the data in and creates a scipy.interpolate.RegularGridInterpolator object
    to be able to interpolate the computed data
    """
    w_values  = np.loadtxt(data_filename, max_rows = 1)
    t0_values = np.loadtxt(data_filename, skiprows = 1, max_rows = 1)
    physical_t0_values = np.loadtxt(data_filename, skiprows = 2, max_rows = len(t0_values))

    unbound_times = np.loadtxt(data_filename, skiprows = 2 + len(t0_values)) - physical_t0_values
    interpolator = interp2d(t0_values, w_values, unbound_times, kind = "cubic")

    return interpolator

def compute_density_evolution(delta0_fraction, w):
    sigma = np.linspace(0, np.pi, 2000)
    flrw = FLRW_Phantom(w = w, H0 = 1)

    delta0 = delta0_fraction/(-flrw.alpha)
    a0 = flrw.a(delta0)

    x0    = np.cos(sigma)
    y0    = np.zeros_like(sigma)
    xdot0 = np.zeros_like(sigma)
    ydot0 = np.cos(sigma)
    epsilon0 = np.ones_like(sigma)
    rho_B0 = np.ones_like(sigma)

    u0 = np.array([x0/a0, y0, xdot0, ydot0/a0, epsilon0/a0, rho_B0])

    unbound_times = []
    string = String(sigma, flrw)

    n_breaks = 3
    delta_start = delta0

    delta_all = []
    rho_B     = []

    for i in range(n_breaks):
        a_start = flrw.a(delta_start)
        init_rho_B = 1 if i == 0 else np.interp(delta_start, delta, rho_B_i)

        u0 = np.array([x0/a_start, y0, xdot0, ydot0/a_start, epsilon0/a_start, np.ones_like(sigma)*init_rho_B])

        delta_unbound, u = string.solve(u0, delta0 = delta_start)
        delta = u[0, :]
        rho_B_i = u[2, :]
        unbound_times.append(delta_unbound)

        for j in range(len(delta) - 1):
            delta_all.append(delta[j])
            rho_B.append(rho_B_i[j])

        delta_start = delta_unbound

    delta_all = np.array(delta_all)

    fig, ax = plt.subplots()
    ax.plot(delta_all*(-flrw.alpha), rho_B, "b-", label = r"$\frac{d\rho_B}{dt'} = -2H'\rho_B + n_B a\int d\epsilon/dt' d\sigma$")

    ax.set_xlabel(r"$\delta'/\delta'_\mathrm{max}$")
    ax.set_ylabel(r"$\rho_B$")

    unbinding_time_interpolator = readData("txtfiles/unbinding_times2.txt")
    densityEvolution = DensityEvolution(w, flrw, unbinding_time_interpolator)

    n_delta = 10000
    delta_max = 1/(-flrw.alpha)
    delta = np.linspace(delta0, delta_all[-1], n_delta)

    rho_B = densityEvolution.solve(rho_B0[0], delta)

    ax.plot(delta*(-flrw.alpha), rho_B, "r-", label = r"$\frac{d\rho_B}{dt'} = - 3H'\rho_B + \rho_B/\tau$")
    ax.plot(np.array(delta_all)*(-flrw.alpha), rho_B0[0]*(flrw.a(delta_all)/a0)**(-3), "k--", label = r"$\rho_B \propto a^{-3}$")
    ax.legend()

    plt.show()

compute_density_evolution(0.35, -1.03)
