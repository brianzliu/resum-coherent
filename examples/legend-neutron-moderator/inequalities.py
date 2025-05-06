from resum.multi_fidelity_gaussian_process import MFGPInequalityConstraints
from resum.utilities import plotting_utils_legend as plotting
import numpy as np

class InequalityConstraints(MFGPInequalityConstraints):
    def __init__(self):
        self.get_inner_radius = plotting.get_inner_radius
        self.get_outer_radius = plotting.get_outer_radius
        self.is_crossed = plotting.is_crossed

    def evaluate(self, x):
        super().evaluate(x)
        delta_x = np.ones(len(x))
        for i, xi in enumerate(x[:, :-1]):
            if self.get_inner_radius(xi) < 90.0:
                delta_x[i] = 0.0
            elif self.get_outer_radius(xi) > 265.0:
                delta_x[i] = 0.0
            elif self.get_outer_radius(xi) - self.get_inner_radius(xi) > 20.0:
                delta_x[i] = 0.0
            elif (
                xi[2] * xi[1] * xi[4]
                > 1.05 * np.pi * (self.get_outer_radius(xi)**2 - self.get_inner_radius(xi)**2)
            ):
                delta_x[i] = 0.0
            elif self.is_crossed(xi):
                delta_x[i] = 0.0
            else:
                delta_x[i] = 1.0
        return delta_x[:, None]