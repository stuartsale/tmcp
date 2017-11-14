
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
from __future__ import print_function, division
import abc
import math
import numpy as np
import scipy.special


class IsmPowerspec(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def PS(self):
        return

    @abc.abstractmethod
    def project(self):
        return

    @abc.abstractmethod
    def param_dict(self):
        return


class SM14Powerspec(IsmPowerspec):
    """ This class holds power-spectra of the form described in
        Sale & Magorrian (2014), i.e.:

        p(k) = R * (|k| L)^{2 \omega} / (1 + (|k| L)^2)^{\gamma/2+\omega} .

        Such power take a Kolmogorov-like form: at k>>1/L

        p(k) \propto |k|^{-\gamma} ,

        but are tapered towards 0 for k<<1/L

        Attributes
        ----------
        gamma : float
            The power law slope of the power-spectrum at
            |k| >> 1/L.
            Must be greater than 0.
        omega : float
            Sets the form of the tapering/rollover of the
            power spectrum at |k| << 1/L .
            Must be greater than 0.
        L : float
            The scale corresponding to the roll-over of the
            power-spectrum.
            In a turbulent conext this corresponds to the outer
            scale, i.e. the scale of energy injection.
            All resulting distance and wavenumbers produced in this
            class will be expressed as multiples of L or 1/L
            respectively.
        R : float
            A normalistaion constant
        param_names : list
            The names of the parameters required to uniquely define the
            instance
    """

    param_names = ["gamma", "omega", "L", "var"]

    def __init__(self, gamma=11/3, omega=0, L=1., var=1.):
        """ __init__(gamma=11/3, omega=0, L=1.)

            Initialise a Sale & Magorrian (2014) power spectrum object.

            Attributes
            ----------
            gamma : float, optional
                The power law slope of the power-spectrum at
                |k| >> 1/L.
                Must be greater than 0.
            omega : float, optional
                Sets the form of the tapering/rollover of the
                power spectrum at |k| << 1/L .
                Must be greater than 0.
            L : float, optional
                The scale corresponding to the roll-over of the
                power-spectrum.
                In a turbulent conext this corresponds to the outer
                scale, i.e. the scale of energy injection.
                All resulting distance and wavenumbers produced in this
                class will be expressed as multiples of L or 1/L
                respectively.
            var : float, optional
                The variance implied by the power-spectrum, i.e. the
                integral of the non-DC component over all wavenumbers
        """

        if gamma < 0:
            raise AttributeError("gamma<=0 implies infinite variance!")
        if omega < 0:
            raise AttributeError("omega<=0 implies infinite variance!")
        if L < 0:
            raise AttributeError("Scale length cannot be negative!")

        self.gamma = gamma
        self.omega = omega
        self.L = L
        self.var = var

        # Normalisation

        self.R = 1 / self.norm_const()

    def norm_const(self):
        """ norm_const()

            Determine the normalisation constant as in eqn 13 of
            Sale & Magorrian (2014)

            Returns
            -------
            R : float
                normalisation constant
        """
        norm_const = 4*math.pi * (scipy.special.beta((self.gamma-3)/2,
                                                     1.5+self.omega)
                                  / (2 * math.pow(self.L, 3)))
        return norm_const

    def fill_correction(self, cube_half_length):
        """ fill_correction(kmax)

            Determine approximately what proportion of the total power
            is contained within a cubic array of maximum wavenumber kmax
            in any direction

            Attributes
            ----------
            cube_half_length : float
                half the width/length/height of the cube

            Returns
            -------
            fill_correction : float
                The (approximate) proportion of the total power contained
                within the array.
        """
        # factor of 1.25 is a fudge for cube -> sphere approximation
        kmax = cube_half_length * 1.25
        fill_correction = self.inner_integral(kmax) / self.var
        return fill_correction

    def inner_integral(self, kmax):
        """ inner_integral(kmax)

            Determines the spherical integral of the power spectrum
            from a radius 0 to kmax.

            Attributes
            ----------
            kmax : float
                the maximum wavenumber

            Returns
            -------
            integral : float
                The spherical integral of the power spectum from 0
                to kmax
        """
        integral = (scipy.special.hyp2f1(1.5 + self.omega,
                                         self.gamma/2 + self.omega,
                                         2.5 + self.omega,
                                         -pow(self.L * kmax, 2))
                    * pow(kmax, 3) * pow(self.L*kmax, 2*self.omega)
                    / (3 + 2*self.omega)) * 4 * math.pi * self.var
        return integral * self.R

    def outer_integral(self, kmin):
        """ outer_integral(kmin)

            Determines the spherical integral of the power spectrum
            from a radius kmin to infinity.

            Attributes
            ----------
            kmin : float
                the minimum wavenumber

            Returns
            -------
            integral : float
                The spherical integral of the power spectum from kmin
                to infinity
        """
        integral = (scipy.special.hyp2f1(self.gamma/2 - 1.5,
                                         self.gamma/2 + self.omega,
                                         self.gamma/2 - 0.5,
                                         -pow(kmin*self.L, -2))
                    * pow(kmin, 3) * pow(kmin*self.L, -self.gamma)
                    / (self.gamma - 3)) * 4 * math.pi * self.var
        return integral * self.R

    def PS(self, k):
        """ PS(k)

            Give the (3D) power spectrum for some wavenumber(s) k

            Attributes
            ----------
            k : int, ndarray
                The wavenumbers for which the power-spectrum is needed
        """

        ps = (self.var * self.R * np.power(k*self.L, 2*self.omega)
              / np.power(1 + np.power(k*self.L, 2), self.gamma/2+self.omega))
        return ps

    def project(self, dist_array, dens_func):
        """ project(dens_array, dist_array)

            project the power-spectrum along an axis, given the desity
            along that axis, that is assumed not to vary in the other
            two directions.

            Paramaters
            ----------
            dist_array : ndarray
                An array of distances. Spacing of sequence should be constant
                and monotonically increasing.
                Distances expressed as multiples of L
            dens_array: ndarray
                Densities at the distances given by dist_array
            Notes
            -----
            This function uses an FFT transform.
        """
        k_array_1D = np.fft.rfftfreq(dist_array.size,
                                     (dist_array[1]-dist_array[0]))

        fourier_dens2 = dens_func.fourier2(k_array_1D)

        kx, kz = np.meshgrid(k_array_1D, k_array_1D)
        k_array = np.sqrt(np.power(kx, 2) + np.power(kz, 2))

        ps = self.PS(k_array)

        # Bodge to account for very thick clouds
        if dist_array[-1]/2 < dens_func.half_width:
            projected_ps = ps[:, 0]*np.pi*dens_func.half_width
        else:
            projected_ps = ((ps[:, 0]*fourier_dens2[0]
                            + 2*np.sum(ps[:, 1:]*fourier_dens2[1:], axis=1))
                            * (k_array_1D[1]-k_array_1D[0]))

        return k_array_1D, projected_ps, fourier_dens2

    def param_dict(self):
        """ param_dict()

            Returns (only) the paramaters needed to uniquely specify
            the power spectrum.
            For SM14Powerspec these parameters are:
            gamma, omega, L.

            Parameters
            ----------
            None

            Returns
            -------
            dict
                The parameters that specify the mean function.
        """
        return {"gamma": self.gamma, "omega": self.omega, "L": self.L,
                "var": self.var}

    def MH_propose(self, proposal_width):
        """ MH_propose(proposal_width)

            Obtain a new SM14Powerspec instance for use in
            MH-MCMC samplers.

            Parameters
            ----------
            proposal_width : dict
                The width of the (Gaussian) proposal distibution
                for each paramater

            Returns
            -------
            new_ps : SM14Powerspec
                The new instance
        """
        cls = self.__class__
        new_params = {}
        old_params = self.param_dict()
        for param in self.param_names:
            new_params[param] = (old_params[param]
                                 + proposal_width[param] * np.random.randn())

        new_ps = cls.__init__(**new_params)

        return new_ps
