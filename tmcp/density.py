import collections
import numpy as np


class DensityFunc(object):
    """ class DensityFunc

        A base class to hold a (1d) mean density function.
    """
    __metaclass__ = abc.ABCMeta

    def censored_grid(self, dists, critical_dens):
        censored_dens = self.__call__(dists)
        censored_dens[censored_dens < critical_dens] = 0.
        return censored_dens

    @abc.abstractmethod
    def limited_mean(self):
        return

    @abc.abstractmethod
    def MH_propose(self):
        return

    @abc.abstractmethod
    def param_dict(self):
        return


class QuadraticDensityFunc(DensityFunc):
    """ class QuadraticDensityFunc

        A class to hold a 1D mean density function that is parametrised
        as a 'quadratic lump', i.e.

        rho(s) = { a (s - mid_dist)^2 + max_dens if |s - mid_dist| < half_width
                 { 0  otherwise

        Attributes
        ----------
        mid_dist : float
            The distance of the mid point of the 'lump'
        max_dens : float
            The maximum density in the lump
        half_width : float
            The half width of the lump
        a : float
            The quadratic coefficiant, derived from max_dens and
            half_width
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

    def MH_propose(self, proposal_width):
        """ MH_propose(proposal_width)

            Obtain a new QuadraticDensityFunc instance for use in
            MH-MCMC samplers.

            Parameters
            ----------
            proposal_width : dict
                The width of the (Gaussian) proposal distibution
                for each paramater

            Returns
            -------
            QuadraticDensityFunc
                The new instance
        """
        new_density_func = cp.deepcopy(self)
        for key in density_prop:
            new_density_func.__dict__[key] = (prev_cloud.density_func
                                              .__dict__[key]
                                              + density_prop[key]
                                              * np.random.randn())
        new_density_func.a = (-new_density_func.max_dens
                              / pow(new_density_func.half_width, 2))
        return new_density_func

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

    def param_dict(self):
        """ param_dict()

            Returns (only) the paramaters needed to uniquely specify
            the density function.
            For QuadraticDensityFunc these parameters are:
            mid_dist, max_dens, half_width.

            Parameters
            ----------
            None

            Returns
            -------
            dict
                The parameters that specify the mean function.
        """

        return {"mid_dist": self.mid_dist, "max_dens": self.max_dens,
                "half_width": self.half_width}
