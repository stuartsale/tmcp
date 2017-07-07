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

    def __init__(self, emitter_abundances, emitter_lines, mean_nH=1e2,
                 cfac=None, depth=100, Reff=1, sigmaNT=2.0e5, Tg=10.,
                 xoH2=0.1, xpH2=0.4, xHe=0.1, min_col=19, max_col=24,
                 steps=11):
        """ __init__(emitter_abundances, emitter_lines, mean_nH=1e2,
                     cfac=None, depth=100, Reff=None, sigmaNT=2.0e5, Tg=10.,
                     xoH2=0.1, xpH2=0.4, xHe=0.1, min_col=19, max_col=24,
                     steps=11)

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
        mean_nH : float
            The mean volume density of H nuclei in cm^-3
        cfac : float
            The clustering factor of the cloud. If Reff is set, this is
            the clustering value *inside* that radius and so will be lower
            than the value for the entire cloud
        depth : float
            The depth of the cloud along the line of sight.
            Measured in parsecs.
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
        self.cloud.nH = mean_nH
        self.cloud.sigmaNT = sigmaNT
        self.cloud.Tg = Tg
        self.cloud.comp.xoH2 = xoH2
        self.cloud.comp.xpH2 = xpH2
        self.cloud.comp.xHe = xHe
        if cfac is not None:
            self.cloud.cfac = cfac
        if Reff is not None:
            self.cloud.Reff = Reff

        self.depth = depth

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
                    TB_dict[emitter][line][i] = lines_dicts[line]["intTB"]

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
