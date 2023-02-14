import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

class FLRW:
    """
    Instances of this class represent different FLRW-metrics with a(t) defined by w and H0.
    The class contains methods for computing a(t) and its time-derivative, as well as H(t).
    Only works for w =/= -1
    """
    def __init__(self, w, H0):
        self.w = w
        self.alpha = 3/2*(1 + self.w)
        self.H0 = H0

        self.t_rip = 2/(3*np.abs(1 + self.w)*self.H0)

    def a(self, t):
        return ((1 + self.alpha*t))**(1/self.alpha)

    def H(self, t):
        return 1/(1 + self.alpha*t)

class String:
    """
    Represents a string moving in the given metric.
    Contains methods for:
        - computing the right-hand-sides (rhs) of the equations of motion for the string
        - computing the update in x, y, xdot, ydot and epsilon by using the 4th order Runge Kutta algorithm
        - solving for the dynamics of the string given initial conditions
        - computing the energy of the string in a given state
    """
    def __init__(self, sigma, metric):
        self.sigma = sigma
        self.metric = metric

    def rhs(self, u, t):
        x, y, xdot, ydot, epsilon, rho_B = u

        a = self.metric.a(t)
        H = self.metric.H(t)

        v = xdot**2 + ydot**2

        # Calculating gradients numerically
        x_grad = np.gradient(x, self.sigma, edge_order = 2)
        y_grad = np.gradient(y, self.sigma, edge_order = 2)
        grad_term_x = 1/a**2*np.gradient(x_grad/epsilon, self.sigma, edge_order = 2)/epsilon
        grad_term_y = 1/a**2*np.gradient(y_grad/epsilon, self.sigma, edge_order = 2)/epsilon

        # Calculating the time derivatives of x, y, xdot, ydot, epsilon, rho_B
        derivatives = np.zeros_like(u)
        derivatives[0, :] = xdot
        derivatives[1, :] = ydot

        derivatives[2, :] = -3*H*xdot + 2*a**2*H*v*xdot + grad_term_x
        derivatives[3, :] = -3*H*ydot + 2*a**2*H*v*ydot + grad_term_y
        derivatives[4, :] = -2*a**2*H*epsilon*v

        E = self.energy(u[4, :], t)
        epsdot_integral = np.trapz(derivatives[4, :], self.sigma)*a

        dEdt = H*E + epsdot_integral

        n_B = rho_B/E
        derivatives[5, :] = -3*H*rho_B + n_B*dEdt

        return derivatives

    def RK4(self, u, t, dt):
        """
        General implementation of the 4th order Runge Kutta algorithm
        """

        k1 = self.rhs(u, t)
        k2 = self.rhs(u + k1*dt/2, t + dt/2)
        k3 = self.rhs(u + k2*dt/2, t + dt/2)
        k4 = self.rhs(u + k3*dt, t + dt)

        return 1/6*(k1 + 2*(k2 + k3) + k4)

    def solve(self, u0, t0):
        """
        Solves for the dynamics of the string. Only need to solve up until the string breaks, which
        is when the current energy of the string exceeds twice its initial energy.
        Need to use adaptive step size, since the appropriate step size depends on how close we are to the big rip.
        """
        E = [self.energy(u0[4, :], t0)] # Want to store energy values, initializing list here
        rho_B = [u0[5, 0]]

        t = [t0]
        u = np.copy(u0)

        t_max = -1/self.metric.alpha

        # Computing the new state of the string until it breaks
        while E[-1] < 2*E[0]:
            dt = (t_max - t[-1])/(t_max*1000)#*np.exp(-(2 + self.metric.w))

            du = self.RK4(u, t[-1], dt)*dt
            u += du

            t.append(t[-1] + dt)
            E.append(self.energy(u[4, :], t[-1]))
            rho_B.append(u[5, 0])

        """
        #print(E)
        fig, ax = plt.subplots()
        ax.plot(t, E, "m-")
        ax.plot([t[0], t[-1]], [2*E[0], 2*E[0]], "k--")

        fig, ax = plt.subplots()
        ax.plot(t, rho_B, "g-")
        ax.plot(t, np.array(rho_B[0])*(self.metric.a(np.array(t))/self.metric.a(t[0]))**(-3), "k--")
        fig.savefig("images/rho_B_explicit.pdf")
        """

        # Using interpolation to make the time of breaking, t_unbound, more accurate
        # Otherwise the accuracy of this would be limited by the stepsize between the last two steps
        t_unbound = np.interp(2*E[0], E, t)
        return t_unbound, np.array(t), np.array(rho_B)

    def energy(self, epsilon, t):
        """
        The energy of a string in a state characterized by epsilon, at time t
        """
        return np.trapz(epsilon, self.sigma)*self.metric.a(t)

class BaryonDensity:
    def __init__(self, w, unbinding_time_interpolator):
        self.w = w
        self.unbinding_time_interpolator = unbinding_time_interpolator
        self.flrw = FLRW(w = w, H0 = 1)

    def rhs(self, rho_B, t):
        return -3*self.flrw.H(t)*rho_B + rho_B/self.unbinding_time_interpolator(t*(-self.flrw.alpha), self.w)[0]

    def RK4(self, rho_B, t, dt):
        k1 = self.rhs(rho_B, t)
        k2 = self.rhs(rho_B + k1*dt/2, t + dt/2)
        k3 = self.rhs(rho_B + k2*dt/2, t + dt/2)
        k4 = self.rhs(rho_B + k3*dt, t + dt)

        return 1/6*(k1 + 2*(k2 + k3) + k4)

    def solve(self, rho0_B, t):
        n = len(t)
        dt = (t[-1] - t[0])/n

        rho_B = np.zeros_like(t)
        rho_B[0] = rho0_B

        for i in range(n - 1):
            rho_B[i + 1] = rho_B[i] + self.RK4(rho_B[i], t[i], dt)*dt

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

if __name__ == "__main__":
    sigma = np.linspace(0, np.pi, 2000)
    H0 = 3.77e-42
    w = -1.03

    flrw = FLRW(w = w, H0 = H0)
    # For w = -1.03 about t0 = 0.35 is the lowest we can go.
    t0 = 0.35/(-flrw.alpha)
    a0 = flrw.a(t0)

    # Initial conditions for an initially linear string along the x-axis
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
    t_start = t0

    t_all = []
    rho_B = []

    for i in range(n_breaks):
        a_start = flrw.a(t_start)
        init_rho_B = 1 if i == 0 else np.interp(t_start, t, rho_B_i)

        u0 = np.array([x0/a_start, y0, xdot0, ydot0/a_start, epsilon0/a_start, np.ones_like(sigma)*init_rho_B])

        t_unbound, t, rho_B_i = string.solve(u0, t0 = t_start)
        unbound_times.append(t_unbound)

        for j in range(len(t) - 1):
            t_all.append(t[j])
            rho_B.append(rho_B_i[j])

        t_start = t_unbound

    fig, ax = plt.subplots()
    #ax.plot(t_all, rho_B, "b-", label = r"$\rho_B\left(t'\right)$")
    ax.plot(np.array(t_all)*(-flrw.alpha), rho_B, "b-", label = r"$\frac{d\rho_B}{dt'} = -2H'\rho_B + n_B a\int d\epsilon/dt' d\sigma$")

    t_all = np.array(t_all)


    ax.set_xlabel(r"$\delta'/\delta'_\mathrm{max}$")
    ax.set_ylabel(r"$\rho_B$")

    unbinding_time_interpolator = readData("txtfiles/unbinding_times2.txt")
    baryonDensity = BaryonDensity(w = w, unbinding_time_interpolator = unbinding_time_interpolator)

    n_t = 10000
    t = np.zeros(n_t)
    t_max = 1/(-baryonDensity.flrw.alpha)

    t = np.linspace(t0, t_all[-1], n_t)
    rho_B = baryonDensity.solve(rho_B0[0], t)

    ax.plot(t*(-flrw.alpha), rho_B, "r-", label = r"$\frac{d\rho_B}{dt'} = - 3H'\rho_B + \rho_B/\tau$")
    ax.plot(np.array(t_all)*(-flrw.alpha), rho_B0[0]*(flrw.a(t_all)/a0)**(-3), "k--", label = r"$\rho_B \propto a^{-3}$")
    ax.legend()
    fig.savefig("images/density_evolution_w_103.pdf", bbox_inches = "tight")

    """
    string = String(sigma, flrw)
    #t_unbound = string.solve(u0, t0 = 0.0/(-flrw.alpha))
    t_unbound, rho_B1 = string.solve(u0, t0 = t0)
    print(t_unbound)

    c = a0/flrw.a(t_unbound)
    u0 = np.array([x0*c, y0, xdot0, ydot0*c, epsilon0*c, np.ones_like(sigma)*rho_B1[-1]])
    t_unbound2, rho_B2 = string.solve(u0, t0 = t_unbound)

    t1 = np.linspace(t0, t_unbound, len(rho_B1))
    t2 = np.linspace(t_unbound, t_unbound2, len(rho_B2))

    #print(rho_B2[-1])

    fig, ax = plt.subplots()
    ax.plot(t1, rho_B1, "b-")
    ax.plot(t2, rho_B2, "r-")
    ax.set_xlabel(r"$\delta'$")
    ax.set_ylabel(r"$\rho_B$")
    """

    plt.show()
