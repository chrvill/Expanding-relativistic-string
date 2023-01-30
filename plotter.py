import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from compute_breaking_time import FLRW

class Plotter:
    def __init__(self, data_filename):
        self.data_filename = data_filename

        self.readData()

    def readData(self):
        """
        Loads all the data in and creates a scipy.interpolate.RegularGridInterpolator object
        to be able to interpolate the computed data
        """
        self.w_values  = np.loadtxt(self.data_filename, max_rows = 1)
        self.t0_values = np.loadtxt(self.data_filename, skiprows = 1, max_rows = 1)
        self.physical_t0_values = np.loadtxt(self.data_filename, skiprows = 2, max_rows = len(self.t0_values))

        self.unbound_times = np.loadtxt(self.data_filename, skiprows = 2 + len(self.t0_values)) - self.physical_t0_values
        self.interpolator = RegularGridInterpolator((self.w_values, self.t0_values), self.unbound_times)

    def get_unbound_time(self, w, t0):
        """
        Returns the interpolated values for the unbinding times for all the combinations
        of the given w and t0 values
        """
        points = np.array(np.meshgrid(w, t0)).T
        return self.interpolator(points)

plotter = Plotter("txtfiles/unbinding_times.txt")

w = np.linspace(-2, -1.03, 100)
t0 = np.linspace(0.62, 0.99, 100)

delta_max = -2/(3*(1 + w))

unbound_times = plotter.get_unbound_time(w, t0)

colormap = "inferno"

fig, ax = plt.subplots()
im = ax.imshow(unbound_times, cmap = colormap, interpolation = "gaussian", extent = [t0[0], t0[-1], w[-1], w[0]])
ax.invert_yaxis()
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"$\delta'_\mathrm{unbound} - \delta'_\mathrm{start}$")
ax.set_xlabel(r"$\delta'_\mathrm{start}/\delta'_\mathrm{max}$")
ax.set_ylabel(r"$w$")
ax.set_aspect("auto")

H = np.zeros_like(unbound_times)
for i, w_i in enumerate(w):
    flrw = FLRW(w = w_i, H0 = 3.77e-42)
    for j, t0_j in enumerate(t0):
        H[i, j] = flrw.H(t0_j/(-flrw.alpha))

fig, ax = plt.subplots()
im = ax.imshow(1/H, cmap = colormap, interpolation = "gaussian", extent = [t0[0], t0[-1], w[-1], w[0]])
ax.invert_yaxis()
cbar = fig.colorbar(im)
cbar.ax.set_ylabel(r"$\delta'_\mathrm{unbound} - \delta'_\mathrm{start}$")
ax.set_xlabel(r"$\delta'_\mathrm{start}/\delta'_\mathrm{max}$")
ax.set_ylabel(r"$w$")
ax.set_aspect("auto")

plt.show()
