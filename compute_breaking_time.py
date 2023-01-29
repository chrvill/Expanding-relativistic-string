import numpy as np
import matplotlib.pyplot as plt

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
        #return 0.5**(-1/self.alpha)*((1 + self.alpha*t))**(1/self.alpha)
        return ((1 + self.alpha*t))**(1/self.alpha)

    def adot(self, t):
        #return self.H0*(1 + self.alpha*self.H0*t)**(1/self.alpha - 1)
        return self.H0*(1 + self.alpha*t)**(1/self.alpha - 1)

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
        x, y, xdot, ydot, epsilon = u

        a = self.metric.a(t)
        H = self.metric.H(t)

        # Need to normalize the velocity to enforce the requirement that
        # the endpoints of the string move at the speed of light.
        # Want the velocity distribution to have the same "shape", just rescaling the whole
        # distribution
        v = xdot**2 + ydot**2
        vel_distrib = v/v[0]

        # Calculating gradients numerically
        x_grad = np.gradient(x, self.sigma, edge_order = 2)
        y_grad = np.gradient(y, self.sigma, edge_order = 2)

        grad_term_x = 1/a**2*np.gradient(x_grad/epsilon, self.sigma, edge_order = 2)/epsilon
        grad_term_y = 1/a**2*np.gradient(y_grad/epsilon, self.sigma, edge_order = 2)/epsilon

        # Calculating the time derivatives of x, y, xdot, ydot, epsilon
        derivatives = np.zeros_like(u)
        derivatives[0, :] = xdot
        derivatives[1, :] = ydot

        derivatives[2, :] = -3*H*xdot + 2*H*xdot*vel_distrib + grad_term_x
        derivatives[3, :] = -3*H*ydot + 2*H*ydot*vel_distrib + grad_term_y
        derivatives[4, :] = -2*H*epsilon*vel_distrib

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
        Need to use adaptive step size, since the appropriate step size varies depends on w and
        only on how close we are to the big rip.
        """
        #t_rip = self.metric.t_rip   # Time of big rip for the w and H0 in question
        E = [self.energy(u0[-1, :], t0)] # Want to store energy values, initializing list here

        t = [t0]
        u = np.copy(u0)

        t_max = -1/self.metric.alpha

        # Computing the new state of the string until it breaks
        while E[-1] < 2*E[0]:
            dt = (t_max - t[-1])/(t_max*1000)#*np.exp(-(2 + self.metric.w))

            du = self.RK4(u, t[-1], dt)*dt
            u += du

            t.append(t[-1] + dt)
            E.append(self.energy(u[-1, :], t[-1]))
            #print(E[-1], np.max(np.abs(du)), self.metric.H(t[-1]))

        fig, ax = plt.subplots()
        ax.plot(t, E, "m-")
        ax.plot([t[0], t[-1]], [2*E[0], 2*E[0]], "k--")
        #plt.show()

        # Using interpolation to make the time of breaking, t_unbound, more accurate
        # Otherwise the accuracy of this would be limited by the stepsize between the last two steps
        t_unbound = np.interp(2*E[0], E, t)
        return t_unbound

    def solve_and_store_dynamics(self, u0, t0):
        E = [self.energy(u0[-1, :], t0)]

        t = [t0]
        u = [u0]

        t_max = -1/self.metric.alpha

        while E[-1] < 2*E[0]:
            dt = (t_max - t[-1])/(t_max*1000)#*np.exp(-(2 + self.metric.w))

            u.append(u[-1] + self.RK4(u[-1], t[-1], dt)*dt)

            t.append(t[-1] + dt)
            E.append(self.energy(np.array(u)[-1, -1, :], t[-1]))

        print(len(t))
        return t, np.array(u), E

    def energy(self, epsilon, t):
        """
        The energy of a string in a state characterized by epsilon, at time t
        """
        return np.trapz(epsilon, self.sigma)*self.metric.a(t)

def compute_unbound_times_standard():
    # The sigma-values we will use throughout
    sigma = np.linspace(0, np.pi, 2000)

    flrw = FLRW(w = -1.03, H0 = 3.77e-42)
    # For w = -1.03 about t0 = 0.61085 is the lowest we can go.
    #t0 = 0.61085/(-flrw.alpha)
    t0 = 0.62/(-flrw.alpha)
    a = flrw.a(t0)

    # Initial conditions for an initially linear string along the x-axis
    x0    = np.cos(sigma)/a
    y0    = np.zeros_like(sigma)
    xdot0 = np.zeros_like(sigma)
    ydot0 = np.cos(sigma)/a
    epsilon0 = np.ones_like(sigma)/a

    u0 = np.array([x0, y0, xdot0, ydot0, epsilon0])

    string = String(sigma, flrw)
    #t_unbound = string.solve(u0, t0 = 0.0/(-flrw.alpha))
    t_unbound = string.solve(u0, t0 = t0)
    print(t_unbound)

    """
    t, u, E = string.solve_and_store_dynamics(u0, t0 = t0)
    fig, ax = plt.subplots()
    ax.plot(u[0, 0, :], u[0, 1, :], "b-")
    ax.plot(u[100, 0, :], u[100, 1, :], "g-")
    ax.plot(u[-1, 0, :], u[-1, 1, :], "r-")
    print(t[-1])
    """
    """
    # The w- and t0-values for which we will compute t_unbound
    # t0-values are given by a fraction x, representing x*t_rip
    w_values  = np.linspace(-2, -1.1, 20)
    t0_values = np.linspace(0, 0.99, 20)

    # The physical times the t0-values actually represent, so t0*t_rip
    physical_t0_values = np.zeros((len(w_values), len(t0_values)))

    # Physical time depends on t_rip, which in turn depends on w.
    # Need to calculate physical time for every combination of t0 and w
    for i, w in enumerate(w_values):
        flrw = FLRW(w = w, H0 = 1)
        for j, t0 in enumerate(t0_values):
            physical_t0_values[i, j] = t0/(-flrw.alpha)
    """
    plt.show()
    """
    # Now we compute t_unbound for every combination of t0 and w
    unbound_times = np.zeros((len(w_values), len(t0_values)))
    for i, w in enumerate(w_values):
        flrw = FLRW(w = w, H0 = 1)
        string = String(sigma, flrw)

        for j, t0 in enumerate(t0_values):
            t_unbound = string.solve(u0, t0 = t0/(-flrw.alpha))
            unbound_times[i, j] = t_unbound

    filename = "txtfiles/unbound_time_map.txt"
    # Saving w, t0 and unbound_times to file
    with open(filename, "w") as file:
        np.savetxt(file, [w_values, t0_values])
        np.savetxt(file, physical_t0_values)
        np.savetxt(file, unbound_times)
    """

if __name__ == "__main__":
    compute_unbound_times_standard()
