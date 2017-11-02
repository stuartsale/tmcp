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
import despotic as dp
import math
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as physcons

from . import density, powerspec

parsec = physcons.parsec*100.  # in cm


class CoGsObj(object):
    """ This class holds Curves of Growth (CoG) that relate real
        column densities to observed brightness temperatures.

        CoGs are calculated using the DESPOTIC library, see
        despotic.readthedocs.io  and Krumholz, 2014, MNRAS, 437, 1662,
        DOI: 10.1093/mnras/stt2000 for more information.

        Attributes
        ----------
        cloud : despotic.cloud
            A cloud object from DESPOTIC
        spline : dict
            Each pair in the dict corresponds to an emitter, where
            the value is itself a dict where the keys correspond to lines
            and the values are InterpolatedUnivariateSpline objects that
            give brightness temperature given some column density
    """

    def __init__(self, emitter_abundances, emitter_lines, dens_func, ps,
                 Reff=1., sigmaNT=2.0e5, Tg=10., xoH2=0.1, xpH2=0.4, xHe=0.1,
                 min_col=19, max_col=24, steps=11):
        """ __init__(emitter_abundances, emitter_lines, dens_func, ps,
                     Reff=None, sigmaNT=2.0e5, Tg=10., xoH2=0.1, xpH2=0.4,
                     xHe=0.1, min_col=19, max_col=24, steps=11)

        Initialise a CoGObj

        Attributes
        ----------
        emitter_abundances : dict
            A dictionary whose keys are species name strings and whose
            values are relative abundances
        emitter_lines : dict
            A dictionary whose keys are species name strings and whose
            values are lists of lines needed (ordered by freq for each
            species.
        dens_func : DensityFunc or derived
            The mean density function of the cloud
        ps : IsmPowerspec or derived
            The power spectrum within the cloud.
        Reff : float
            The effective radius of the cloud used when estimating
            escape probabilities. Default is None, which implies
            size set by column density and nH.
        sigmaNT : float
            Non-thermal velocity dispersion
        Tg : float
            Gas temperature in Kelvin
        xoH2 : float
            relative abundance of ortho-H2
        xpH2 : float
            relative abundance of para-H2
        xHe : float
            relative abundance of He
        min_col : float
            The minimum log10 of the column density of H nuclei in cm^-2
            to be used
        max_col : float
            The maximum log10 of the column density of H nuclei in cm^-2
            to be used
        steps : int
            The number of steps used when finding the CoG
        """

        # setup cloud

        self.cloud = dp.cloud()

        # check dens_func and ps types
        if not isinstance(dens_func, density.UniformDensityFunc):
            raise NotImplementedError("Currently only implemented with "
                                      "UniformDensityFunc density function "
                                      "instances")
        if not isinstance(ps, powerspec.SM14Powerspec):
            raise NotImplementedError("Currently only implemented with"
                                      "SM14Powerspec power-spectrum instances")

        self.dens_func = dens_func
        self.ps = ps

        self.cloud.sigmaNT = sigmaNT
        self.cloud.Tg = Tg
        self.cloud.comp.xoH2 = xoH2
        self.cloud.comp.xpH2 = xpH2
        self.cloud.comp.xHe = xHe

        # Set some derived cloud params

        self.cloud.nH = self.dens_func.dens_0
        self.depth = self.dens_func.half_width * 2.
        self.cloud.colDen = self.dens_func.integral() * parsec

        if Reff is not None:
            self.cloud.Reff = Reff
        else:
            self.cloud.Reff = self.depth

        var_R = ps.outer_integral(1./self.cloud.Reff)

        self.cloud.cfac = var_R/pow(self.cloud.nH, 2) + 1.

        # add emitters

        for emitter in emitter_abundances:
            self.cloud.addEmitter(emitter, emitter_abundances[emitter])

        # set up dicts and arrays needed

        cols = np.linspace(min_col, max_col, steps)

        TB_dict = {}
        emitter_trans = {}
        for emitter in emitter_lines:
            TB_dict[emitter] = {}
            emitter_trans[emitter] = [np.array(emitter_lines[emitter])+1,
                                      np.array(emitter_lines[emitter])]
            for line in emitter_lines[emitter]:
                TB_dict[emitter][line] = np.zeros(steps)

        # Find values

        for i, col in enumerate(cols):
            self.cloud.colDen = math.pow(10, col)
            self.cloud.nH = self.cloud.colDen / (self.depth * parsec)
            for emitter in emitter_lines:
                lines_dicts = self.cloud.lineLum(
                                            emitter, kt07=True,
                                            transition=emitter_trans[emitter])
                for line in emitter_lines[emitter]:
                    TB_dict[emitter][line][i] = (
                            lines_dicts[emitter_lines[emitter].index(line)]
                            ["intTB"])

        # Fit splines

        self.splines = {}
        for emitter in emitter_lines:
            self.splines[emitter] = {}
            for line in emitter_lines[emitter]:
                self.splines[emitter][line] = (
                    InterpolatedUnivariateSpline(cols, TB_dict[emitter][line]))

    def __call__(self, emitter, line, log_col_dens):
        """ __call__(emitter, line, log_col_dens)

            Get the brightness temperature implied for a given line
            by some column density

            Attributes
            ----------
            emitter : string
                The emitting species
            line : int
                the line number (based on the order by freq)
            col_dens : float, ndarray(float)
                The log10 of column density of H nucleons in cm^-2

            Returns
            -------
            TB : float
                The implied brightness temperature(s) given the species
                and line.
        """

        # Raises KeyError if species and line have been prepared

        try:
            return self.splines[emitter][line](log_col_dens)
        except KeyError:
            if emitter not in self.splines.keys():
                raise KeyError("Emitter {0} not in set of CoGs"
                               .format(emitter))
            elif line not in self.splines[emitter].keys():
                raise KeyError("Line {0} for emitter {1} not in set of CoGs"
                               .format(line, emitter))
