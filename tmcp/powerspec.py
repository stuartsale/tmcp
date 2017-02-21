from __future__ import print_function, division
import math
import numpy as np
import scipy.special


class ISM_powerspec(object):
    pass


class SM14_powerspec(ISM_powerspec):
    """ This class holds power-spectra of the form described in
        Sale & Magorrian (2014), i.e.:

        p(k) = R(k) * (|k| L)^{2 \omega} / (1 + (|k| L)^2)^{\gamma/2+\omega} .

        Such power take a Kolmogorov-like form: at k>>1/L

        p(k) \propto |k|^{-\gamma} ,

        but are tapered towards 0 for k<<1/L
    """

    def __init__(self, gamma=11/3, omega=0, L=1.):
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

        # Normalisation

        norm_const = 4*math.pi * ((scipy.special.gamma((gamma-3)/2)
                                   * scipy.special.gamma(1.5+omega))
                                  / (2 * scipy.special.gamma(gamma/2+omega)))
        self.R = 1 / norm_const

    def PS(self, k):
        """ PS(k)

            Give the (3D) power spectrum for some wavenumber(s) k

            Attributes
            ----------
            k : int, ndarray
                The wavenumbers for which the power-spectrum is needed
        """

        ps = (self.R * np.power(k*self.L, 2*self.omega)
              / np.power(1 + np.power(k*self.L, 2), self.gamma/2+self.omega))
        return ps

    def project(self, dist_array, dens_array):
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

        fourier_dens = np.fft.rfft(dens_array)
        fourier_dens2 = (fourier_dens*fourier_dens.conj()).real

        k_array_1D = np.fft.rfftfreq(dens_array.size,
                                     (dist_array[1]-dist_array[0]))
        kx, ky = np.meshgrid(k_array_1D, k_array_1D)
        k_array = np.sqrt(np.power(kx, 2) + np.power(ky, 2))

        ps = self.PS(k_array)

        projected_ps = ((ps[:, 0]*fourier_dens2[0]
                         + 2*np.sum(ps[:, 1:]*fourier_dens2[1:], axis=1))
                        * (k_array_1D[1]-k_array_1D[0])/4)

        return k_array_1D, projected_ps, fourier_dens
