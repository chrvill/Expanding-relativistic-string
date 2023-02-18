import numpy as np
import matplotlib.pyplot as plt

class FLRW:
    def __init__(self, w, H0):
        self.w = w
        self.H0 = H0
        self.alpha = 3/2*(1 + self.w)

class FLRW_Phantom(FLRW):
    def __init__(self, w, H0):
        FLRW.__init__(self, w, H0)

        self.t_rip = 2/(3*np.abs(1 + self.w)*self.H0)

    def a(self, delta):
        return (1 + self.alpha*delta)**(1/self.alpha)

    def H(self, delta):
        return 1/(1 + self.alpha*delta)

class FLRW_general(FLRW):
    def __init__(self, w, H0, rho_crit, omega_phantom0):
        FLRW.__init__(self, w, H0)
        self.rho_crit = rho_crit
        self.omega_phantom0 = omega_phantom0

class String:
    def __init__(self, sigma, metric):
        self.sigma = sigma
        self.metric = metric

    def rhs_phantom(self, u, delta):
        x, y, xdot, ydot, epsilon, rho_B = u

        a = self.metric.a(delta)
        H = self.metric.H(delta)

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

        E = self.energy(u[4, :], a)
        epsdot_integral = np.trapz(derivatives[4, :], self.sigma)*a
        dEdt = H*E + epsdot_integral
        n_B = rho_B/E
        derivatives[5, :] = -3*H*rho_B + n_B*dEdt

        return derivatives

    def rhs_general(self, u, delta):
        x, y, xdot, ydot, epsilon, rho_B, a = u

        H = self.metric.H0*np.sqrt(rho_B/self.metric.rho_crit \
            + self.metric.omega_phantom0*a**(-3*(1 + self.metric.w)))
        #print(self.metric.H0)

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

    def RK4(self, u, delta, ddelta, rhs_func):
        k1 = rhs_func(u, delta)
        k2 = rhs_func(u + k1*ddelta/2, delta + ddelta/2)
        k3 = rhs_func(u + k2*ddelta/2, delta + ddelta/2)
        k4 = rhs_func(u + k3*ddelta, delta + ddelta)

        return 1/6*(k1 + 2*(k2 + k3) + k4)

    def solve(self, u0, delta0, type = "phantom"):
        rho_B = [u0[5, 0]]

        a = []
        if type == "phantom":
            a.append(self.metric.a(delta0))
            rhs_func = self.rhs_phantom
        else:
            a.append(u0[6, 0])
            rhs_func = self.rhs_general

        E = [self.energy(u0[4, :], a[0])]
        delta = [delta0]
        u = np.copy(u0)

        delta_max = -1/self.metric.alpha

        while E[-1] < 2*E[0]:
            if type == "phantom":
                ddelta = (delta_max - delta[-1])/(delta_max*1000)#*np.exp(-(2 + self.metric.w))
            else:
                ddelta = 1e-4

            du = self.RK4(u, delta[-1], ddelta, rhs_func)*ddelta
            u += du

            delta.append(delta[-1] + ddelta)
            #print(delta[-1])

            if type == "phantom":
                a.append(self.metric.a(delta[-1]))
            else:
                a.append(u[6, 0])

            E.append(self.energy(u[4, :], a[-1]))
            rho_B.append(u[5, 0])

        delta_unbound = np.interp(2*E[0], E, delta)
        u = np.array([delta, E, rho_B, a])

        return delta_unbound, u

    def energy(self, epsilon, a):
        return np.trapz(epsilon, self.sigma)*a
