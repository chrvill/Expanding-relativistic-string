import numpy as np
import matplotlib.pyplot as plt
from breaking_with_density_evolution import String

H0 = 3
a0 = 1

#omega_m0 = 0.9999
omega_m0 = 0.99
omega_w0 = 1 - omega_m0

rho_crit = 1

rho_B0 = omega_m0*rho_crit

w = -1.03

class FLRW:
    def __init__(self, w, H0):
        self.w = w
        self.H0 = H0
        self.alpha = 3/2*(1 + self.w)


"""
class FLRW:
    def a(self, t):
        return (1 + 3/2*H0*t)**(2/3)

    def H(self, t):
        return H0/(1 + 3/2*H0*t)
"""


class EdS_String(String):
    def __init__(self, sigma, metric):
        String.__init__(self, sigma, metric)

    def rhs(self, u, t):
        x, y, xdot, ydot, epsilon, rho_B, a = u

        H = H0*np.sqrt(rho_B/rho_crit + omega_w0*(a/a0)**(-3*(1 + w)))

        v = xdot**2 + ydot**2

        # Calculating gradients numerically
        x_grad = np.gradient(x, self.sigma, edge_order = 2)
        y_grad = np.gradient(y, self.sigma, edge_order = 2)

        grad_term_x = 1/a**2*np.gradient(x_grad/epsilon, self.sigma, edge_order = 2)/epsilon
        grad_term_y = 1/a**2*np.gradient(y_grad/epsilon, self.sigma, edge_order = 2)/epsilon

        # Calculating the time derivatives of x, y, xdot, ydot, epsilon
        derivatives = np.zeros_like(u)
        derivatives[0, :] = xdot
        derivatives[1, :] = ydot

        derivatives[2, :] = -3*H*xdot + 2*a**2*H*v*xdot + grad_term_x
        derivatives[3, :] = -3*H*ydot + 2*a**2*H*v*ydot + grad_term_y
        derivatives[4, :] = -2*a**2*H*epsilon*v

        E = self.energy(epsilon, a[0])
        epsdot_integral = np.trapz(derivatives[4, :], self.sigma)*a

        dEdt = H*E + epsdot_integral

        n_B = rho_B/E
        derivatives[5, :] = -3*H*rho_B + n_B*dEdt
        derivatives[6, :] = a*H

        return derivatives

    def solve(self, u0, t0):
        """
        Solves for the dynamics of the string. Only need to solve up until the string breaks, which
        is when the current energy of the string exceeds twice its initial energy.
        Need to use adaptive step size, since the appropriate step size varies depends on w and
        only on how close we are to the big rip.
        """
        #t_rip = self.metric.t_rip   # Time of big rip for the w and H0 in question
        E = [self.energy(u0[4, :], u0[6, 0])] # Want to store energy values, initializing list here
        rho_B = [u0[5, 0]]
        a = [u0[-1, 0]]

        t = [t0]
        u = np.copy(u0)

        dt = 1e-4
        # Computing the new state of the string until it breaks
        while E[-1] < 2*E[0]:
            du = self.RK4(u, t[-1], dt)*dt
            u += du

            a.append(u[6, 0])
            t.append(t[-1] + dt)
            E.append(self.energy(u[4, :], a[-1]))
            rho_B.append(u[5, 0])

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
        fig.savefig("images/a(t)_w_103_omegam0_099.pdf", bbox_inches = "tight")

        a_EdS = a[0]*(1 + 3/2*H0*(np.array(t) - t[0]))**(2/3)

        ax.plot(t, a_EdS, "k--", label = r"$a_\mathrm{EdS} \propto \left(t'\right)^{2/3}$")
        ax.legend()

        fig, ax = plt.subplots()
        ax.plot(t, rho_B/rho_B[0], "r-", label = r"$\rho_B(t')$")
        ax.set_xlabel(r"$t'$")
        ax.set_ylabel(r"$\rho_B/\rho_{B0}$")
        ax.set_title(r"$\Omega_{m0} = $ %s, $w = $ %s" % (omega_m0, w))
        fig.savefig("images/density_general_w_103_omegam0_099.pdf", bbox_inches = "tight")

        ax.plot(t, np.array(a)**(-3), "k--", label = r"$\rho_B \propto a^{-3}$")
        ax.legend()
        plt.show()

        t_unbound = np.interp(2*E[0], E, t)
        return t_unbound, rho_B

    def energy(self, epsilon, a):
        """
        The energy of a string in a state characterized by epsilon, at time t
        """
        return np.trapz(epsilon, self.sigma)*a

sigma = np.linspace(0, np.pi, 2000)

flrw = FLRW(w, H0)
string = EdS_String(sigma, flrw)

t0 = 0
#a0 = flrw.a(t0)

# Initial conditions for an initially linear string along the x-axis
x0    = np.cos(sigma)/a0
y0    = np.zeros_like(sigma)
xdot0 = np.zeros_like(sigma)
ydot0 = np.cos(sigma)/a0
epsilon0 = np.ones_like(sigma)/a0
rho0_B = np.ones_like(sigma)*rho_B0
a0 = np.ones_like(sigma)*a0

u0 = np.array([x0, y0, xdot0, ydot0, epsilon0, rho0_B, a0])

#t_unbound = string.solve(u0, t0 = 0.0/(-flrw.alpha))
t_unbound, rho_B = string.solve(u0, t0 = t0)
print(rho_B[-1]/rho_B[0])
print(t_unbound)
