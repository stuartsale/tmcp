""" Module that performs fit of LRF clouds (& their params)
    to some observed cloud. The fit is based solely on the
    histograms of the observed brightness temperatures
    (thus not accounting for info in covariance function, etc).

    The fit is performed using Approximate Bayesian Computation (ABC)
"""
from __future__ import print_function, division
import math
import numpy as np
from scipy.stats import ks_2samp
from simpleabc import Model, basic_abc, pmc_abc

from lrf_cloud import LRFCloud

outer_L = 64

# ============================================================================


def normpdf(x, mu, sigma):
    u = (x - mu) / sigma
    y = (1 / (math.sqrt(2*math.pi) * sigma)) * np.exp(-u*u/2)
    return np.prod(y)


class BasicCloudPrior():
    """ A class containing a basic prior on the parameters of a cloud.
        The applied prior is a shifted multivariate lognormal with
        0 in the off diagonal elements of the covariance matrix.

        Parameters
        ----------
        None

        Attributes
        ----------
        None
    """
    __mean = np.array([1000., 100., 20., 8., 12.])
    __sd = np.array([500., 30., 20., 4., 4.])
    __lower_bound = np.array([100., 10., 5., 1., 3.])

    __mean_dash = __mean - __lower_bound
    __sigma2 = np.log(1 + np.power(__sd/__mean_dash, 2))
    __sigma = np.sqrt(__sigma2)
    __mu = np.log(__mean_dash) - __sigma2/2.

    def __init__(self):
        pass

    def rvs(self):
        z = np.random.randn(5)
        result = (np.exp(BasicCloudPrior.__mu + z * BasicCloudPrior.__sigma)
                  + BasicCloudPrior.__lower_bound)
        return result

    def pdf(self, value):
        z = np.log(value - BasicCloudPrior.__lower_bound)
        prob = normpdf(z, BasicCloudPrior.__mu, BasicCloudPrior.__sigma)

        return prob


class CloudABC(Model):
    """ This class provides the ABC model to be fit to the data.

        This class over-rides the Model class from simpleabc

        Here theta is the variable that holds the cloud parameters
        they are stored as:
        [distance, mean_density, depth, cfac, Tg]

        Parameters
        ----------
        sim_cube_half_length : int
            The size of the simulated cubes
        lines : dict
            The lines for which images are available
        abuns : dict
            The abundances of the species for which we have images

        Attributes
        ----------
    """
    def __init__(self, sim_cube_half_length, lines, abuns):
        self.sim_cube_half_length = sim_cube_half_length
        self.lines = lines
        self.abuns = abuns

        # make list of line_ids
        self.line_ids = []
        for species in self.lines:
            for line in self.lines[species]:
                self.line_ids.append((species, line))

    def draw_theta(self):
        """ Apply a simple truncated normal prior on cloud params
        """
        theta = self.prior.rvs()
        return theta

    def generate_data(self, theta):
        """ Use the LRFCloud class to generate synthetic fBM clouds
        """
        # Generate cloud
        if np.isnan(self.prior.pdf(theta)):
            images = {}
            for line_id in self.line_ids:
                images[line_id] = (np.zeros(pow(self.sim_cube_half_length, 2))
                                   - np.inf)

        else:
            try:
                cloud = LRFCloud(self.sim_cube_half_length, theta[0], theta[1],
                                 theta[2], theta[3], outer_L, theta[4],
                                 self.lines, self.abuns)

                # Generate images
                images = {}
                for line_id in self.line_ids:
                    images[line_id] = cloud.image(line_id[0],
                                                  line_id[1]).flatten()

            except AttributeError:
                images = {}
                for line_id in self.line_ids:
                    images[line_id] = (np.zeros(pow(self.sim_cube_half_length,
                                                    2)) - np.inf)

        return images

    def summary_stats(self, data):
        """ Simply pass the images - the distance method will
            find KS distances
        """
        return data

    def distance_function(self, obs_images, synth_images):
        """ Use the mean KS stat between images as the distance
        """
        N = 0
        KS_sum = 0.
        for image_id in obs_images.keys():
            KS_sum += ks_2samp(obs_images[image_id].flatten(),
                               synth_images[image_id].flatten())[0]
            N += 1

        return KS_sum / N


class ABCFit(object):
    """ This is a class used to fit the parameters of a cloud
        based solely on the 1D pdf of it observations in some
        band(s).

        Approximate Bayesian Computation (ABC) is employed
        to sample from the posterior.

        Parameters
        ----------

        Attributes
        ----------
    """
    def __init__(self, data, sim_cube_half_length, lines, abuns):
        self.sim_cube_half_length = sim_cube_half_length
        self.lines = lines
        self.abuns = abuns

        self.data = data

        self.model = CloudABC(self.sim_cube_half_length, self.lines,
                              self.abuns)

        self.model.set_prior(BasicCloudPrior())
        self.model.set_data = self.data

    def fit(self, method="pmc_abc", epsilon=0.5, min_samples=100, **kwarg):
        """ fit(method="pmc_abc", epsilon=0.5, min_samples=100)

            Use ABC to approximately sample from the posterior.

            Parameters
            ----------
            method : str, optional
                Which ABC method is employed, choices are "basic_abc"
                and "pmc_abc".
            epsilon : float, optional
                The value of epsilon for "basic_abc" and the initial
                value of epsilon for "pmc_abc"
            min_samples : int, optional
                The minimum number of samples required

            Returns
            -------
            None
        """
        if method == "pmc_abc":
            self.posterior_samples = pmc_abc(self.model, self.data,
                                             epsilon_0=epsilon,
                                             min_samples=min_samples, steps=10,
                                             **kwarg)
        elif method == "basic_abc":
            self.posterior_samples = basic_abc(self.model, self.data,
                                               min_samples=min_samples,
                                               epsilon=epsilon, **kwarg)
