""" Contains the function used to create a LRF (i.e. fBM)
    simulated cloud
"""

from __future__ import print_function, division
import math
import numpy as np
import scipy.constants as physcons

import LRF_gen as lg
from tmcp.density import UniformDensityFunc
from tmcp.powerspec import SM14Powerspec
from tmcp.cogs import CoGsObj


parsec = physcons.parsec*100.  # in cm

# ============================================================================


class LRFCloud(object):
    """ Contains a simulated LRF cloud that can be 'observed'
        in different lines.

        Parameters
        ----------
        cube_half_length : int
            Half the length in pixels of the simulated LRF cube.
            Should be an integrer power of 2
        distance : float
            The mean distance of the simulated cloud in pc
        mean_density : float
            The mean gas density in cm^-3
        depth : float
            The depth of the cloud in pc
        cfac : float
            The clumping factor of the cloud, i.e. <n^2>/<n>^2 .
            Must be greater than 1.
        outer_L : float
            The outer scale of turbulence in pc
        Tg : float
            The gas temperature throughout the cloud, expressed in K
        lines : dict(list)
            A dict of lists - the keys are the species and the lists
            specify which line of the species we want
        abuns : dict
            A dict that specifies the abundances of the different species
            in (species, abundance) pairs
        verbose : bool, optional
            If True, some information is dumped to stdout
        pixel_scale : float, optional
            The linear size of each pixel in pc

        Attributes
        ----------
    """

    def __init__(self, cube_half_length, distance, mean_density, depth, cfac,
                 outer_L, Tg, lines, abuns, pixel_scale=1, verbose=False):
        # Check and set parameters

        self.cube_power = int(math.log(cube_half_length*2)/math.log(2))
        self.cube_half_length = pow(2, self.cube_power)

        if distance < depth/2. or depth <= 0 or mean_density <= 0:
            raise AttributeError("The supplied density parameters do not "
                                 "make physical sense: distance {0} pc; "
                                 "mean_density {1} cm^-3; depth {2} pc"
                                 .format(distance, mean_density, depth))
        else:
            self.dens_func = UniformDensityFunc(distance, mean_density,
                                                depth/2.)

        if cfac >= 1:
            self.cfac = cfac
        else:
            raise AttributeError("Supplied cfac: {0} is less than unity"
                                 .format(cfac))
        self.variance = pow(mean_density, 2)*(cfac-1)
        self.outer_L = outer_L

        self.Tg = Tg
        self.pixel_scale = pixel_scale

        self._images = {}

        # CoG
        self.ps = SM14Powerspec(L=outer_L, var=self.variance)
        self.CoG = CoGsObj(abuns, lines, self.dens_func, self.ps, Tg=Tg,
                           min_col=19, max_col=24, steps=11,
                           Reff=outer_L/(2*math.pi))

        if verbose:
            print("cloud cfac:", self.CoG.cloud.cfac)

        # LRF
        outer = self.outer_L / (self.cube_half_length * self.pixel_scale)

        dens_field = lg.LRF_cube(self.cube_power, 1., self.ps.gamma,
                                 "FFT", omega=self.ps.omega, outer=outer).cube

        # Normalise field
        log_dens_field = np.log(dens_field)
        log_dens_field = ((log_dens_field-np.mean(log_dens_field))
                          / np.std(log_dens_field))

        # Impose mean & cfac
        log_dens_field *= math.sqrt(math.log(self.cfac))
        log_dens_field += (math.log(self.dens_func.dens_0)
                           - math.log(self.cfac)/2.)
        self.dens_field = np.exp(log_dens_field)

        if verbose:
            print("mean density", np.mean(self.dens_field))
            print("measured clumping factor",
                  math.pow(np.std(self.dens_field)
                           / np.mean(self.dens_field), 2) + 1)

        # Get column density
        if (self.dens_func.half_width / self.pixel_scale
                > self.cube_half_length):
            raise AttributeError("Cube is not large enough")

        depth_pixel = int(2 * self.dens_func.half_width / self.pixel_scale)
        col_field = np.sum(self.dens_field[:, :, :depth_pixel],
                           axis=-1)[:cube_half_length, :cube_half_length]
        self.log_col_field = np.log10(col_field) + math.log10(parsec)

        if verbose:
            print("mean column density", np.mean(self.log_col_field))
            print("column clumping factor",
                  math.pow(np.std(col_field) / np.mean(col_field), 2) + 1)

    def image(self, species, line):
        """ image(species, line)

            Get the Brightness temperature image for a given spectral
            line.

            Parameters
            ----------
            species : string
                The name of the species whose line we want
            line : int
                The index of the line transition, numbering up from
                ground state - the convention is the same as in
                despotic

            Returns
            -------
            output_image : ndarray
                The image in the required band as brightness temperature
                in K
        """
        line_id = (species, line)
        if line_id in self._images:
            return self._images[line_id]

        else:
            try:
                output_image = self.CoG(species, line, self.log_col_field)
            except KeyError:
                raise KeyError("No CoG available for species {0}, "
                               "line {1}".format(species, line))

            self._images[line_id] = output_image
            return output_image
