from __future__ import print_function, division
import despotic as dp
import math
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


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

    def __init__(self, emitter_abundances, emitter_lines, nH=1e2,
                 sigmaNT=2.0e5, Tg=10., xoH2=0.1, xpH2=0.4, xHe=0.1,
                 min_col=20, max_col=24, steps=10):
        """ __init__(emitters, sigmaNT=2.0e5, Tg=10., xoH2=0.1, xpH2=0.4,
                 xHe=0.1)

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
        nH : float
            The volume density of H nuclei in cm^-3
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

        self.mycloud = dp.cloud()
        self.mycloud.nH = nH
        self.mycloud.sigmaNT = sigmaNT
        self.mycloud.Tg = Tg
        self.mycloud.comp.xoH2 = xoH2
        self.mycloud.comp.xpH2 = xpH2
        self.mycloud.comp.xHe = xHe

        # add emitters

        for emitter in emitter_abundances:
            mycloud.addEmitter(emitter, emitter_abundances[emitter])

        # set up dicts and arrays needed

        cols = np.linspace(min_col, max_col, steps)

        TB_dict = {}
        for emitter in emitter_lines:
            TB_dict[emitter] = {}
            for line in emitter_lines[emitter]:
                TB_dict[emitter][line] = np.zeros(steps)

        # Find values

        for i, col in enumerate(cols):
            self.mycloud.colDen = col
            for emitter in emitter_lines:
                lines_dicts = self.mycloud.lineLum(emitter)
                for line in emitter_lines[emitter]:
                    TB_dict[emitter][line][i] = lines_dicts[line]["intTB"]

        # Fit splines

        self.splines = {}
        for emitter in emitter_lines:
            self.splines[emitter] = {}
            for line in emitter_lines[emitter]:
                self.splines[emitter][line] = (
                    InterpolatedUnivariateSpline(cols, TB_dict[emitter][line]))
