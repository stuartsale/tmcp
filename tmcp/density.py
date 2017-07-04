########################################################################
# Copyright 2017 Stuart Sale
#
# This file is part of tmcp.
#
# tmcp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this tmcp.  If not, see <http://www.gnu.org/licenses/>.
########################################################################

import abc
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

    def MH_propose(self, density_prop):
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
            new_density_func : DensityFunc
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

    @abc.abstractmethod
    def param_dict(self):
        return

    @abc.abstractmethod
    def integral(self):
        return

    @abc.abstractmethod
    def fourier2(self):
        return

class UniformDensityFunc(DensityFunc):
    """ A class to hold a 1D mean density function that is parametrised
        as a 'uniform slab', i.e.

        rho(s) = { rho_0    if |s - mid_dist| < half_width
                 { 0        otherwise

        Attributes
        ----------
        mid_dist : float
            The distance of the mid point of the 'slab'
        dens_0 : float
            The density in the slab
        half_width : float
            The half width of the slab
        param_names : list
            The names of the parameters required to uniquely define the
            instance
    """

    param_names = ["mid_dist", "dens_0", "half_width"]

    def __init__(self, mid_dist, dens_0, width):
        self.mid_dist = mid_dist
        self.dens_0 = dens_0
        self.half_width = width/2.

    def __call__(self, dist):
        if isinstance(dist, np.ndarray):
            dens = np.zeros(dist.shape)
            dens[np.fabs(dist - self.mid_dist) <= self.half_width] = (
                                                                self.dens_0)
            return dens
        else:
            if abs(dist - self.mid_dist) <= self.half_width:
                return self.dens_0
            else:
                return 0.

    def limited_mean(self):
        """ limited_mean()

            Find the mean density in the region where density > 0

            Returns
            -------
            mean density
        """
        return self.dens_0

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

        return {"mid_dist": self.mid_dist, "dens_0": self.dens_0,
                "half_width": self.half_width}

    def integral(self):
        """ integral()

            Calculate the integral of mean density wrt distance, i.e.
            mean column density.

            Note that this function makes no attempt to respect/simplify
            units - the result will be in [density unit] [distance unit],
            thus units like cm^-3 pc are quite possible.

            Returns
            -------
            integral :float
                the integral of mean density wrt distance from 0 to infinity
        """
        return self.dens_0 * 2 * self.half_width

    def fourier2(self, k_array):
        """ fourier2(k_array)

            Return the magnitude squared of the fourier transform
            of the density function at the wavenumbers given by k_array.
            i.e. |FT(dens_func)|^2

            Parameters
            ----------
            k_array : ndarray
                The wavenumber array

            Returns
            -------
            f2_array : ndarray
                The value of the magnitude squared of the fourier 
                transform of the density function at the wavenumber
                given in k_array
        """
        f2_array = pow(self.dens_0 * np.sinc(k_array*self.half_width)
                       * 2 * self.half_width, 2)
        return f2_array

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
        param_names : list
            The names of the parameters required to uniquely define the
            instance
    """

    param_names = ["mid_dist", "max_dens", "half_width"]

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
