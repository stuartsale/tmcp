import collections
import numpy as np


class DensityFunc(object):
    """
    """
    def censored_grid(self, dists, critical_dens):
        censored_dens = self.__call__(dists)
        censored_dens[censored_dens < critical_dens] = 0.
        return censored_dens


class QuadraticDensityFunc(DensityFunc):
    """
    """
    def __init__(self, mid_dist, max_dens, width):
        self.mid_dist = mid_dist
        self.max_dens = max_dens
        self.half_width = width/2.
        self.a = -self.max_dens / pow(self.half_width, 2)

    def __call__(self, dist):
        if isinstance(dist, collections.Sequence):
            dens = self.a*np.power(dist, 2) + self.max_dens
            dens[dens < 0] = 0.
        else:
            dens = max(0., self.a*pow(dist, 2) + self.max_dens)
        return dens

    def limited_mean(self):
        """ limited_mean()

            Find the mean density in the region where density > 0

            Returns
            -------
            mean density
        """
        mean_dens = (2.*self.half_width*self.max_dens
                     + 2.*self.a*pow(self.half_width, 3)/3.)
        return mean_dens
