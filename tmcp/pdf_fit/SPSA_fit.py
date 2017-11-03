""" Module that performs fit of LRF clouds (& their params)
    to some observed cloud. The fit is based solely on the
    histograms of the observed brightness temperatures
    (thus not accounting for info in covariance function, etc).

    The fit is performed using the Simultaneous perturbation
    stochastic approximation algorithm.
"""

import math
import numpy as np

from lrf_cloud import LRFCloud

# ============================================================================


class SPSACloudFit(object):
    """ This is a class used to fit the parameters of a cloud
        based solely on the 1D pdf of it observations in some
        band(s).

        The Simultaneous perturbation stochastic approximation
        algorithm is employed to search for a set of parameters that 
        provides the minimum chi2 for the fit between simulations & 
        the observations.

        Parameters
        ----------

        Attributes
        ----------
    """

    def __init__(self, lines, images, sim_cube_half_length=64
                 start_params = {'distance': 1000., 'mean_density': 100.,
                                 'depth': 50., 'cfac':5., 
                                 'outer_L': 50, 'Tg': 12.}):

        self.abuns = {}
        self.lines = {}

        self.images = {}

        self.params = start_params

        self.sim_cube_half_length = sim_cube_half_length
