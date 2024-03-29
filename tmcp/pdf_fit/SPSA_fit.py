""" Module that performs fit of LRF clouds (& their params)
    to some observed cloud. The fit is based solely on the
    histograms of the observed brightness temperatures
    (thus not accounting for info in covariance function, etc).

    The fit is performed using the Simultaneous perturbation
    stochastic approximation algorithm.
"""
from __future__ import print_function, division
import math
import numpy as np
from scipy.stats import ks_2samp

from lrf_cloud import LRFCloud

outer_L = 64

# ============================================================================


class SPSACloudFit(object):
    """ This is a class used to fit the parameters of a cloud
        based solely on the 1D pdf of it observations in some
        band(s).

        The Simultaneous perturbation stochastic approximation
        algorithm is employed to search for a set of parameters that
        provides the minimum chi2 for the fit between simulations &
        the observations.

        In general SPSA attempts to find the minima of f(x) and
        proceedes by:

        x_{n+1} = x_n - a_n D(x_n)

        Where for dimension i of x:

        D(x_n)_i = (f(x_n + c_n Delta_n) - f(x_ - c_n Delta_n))
                   / 2 c_n Delta_n_i

        where a_n and c_n are floats and Delta_n is a random
        perturbation vector.

        Parameters
        ----------

        Attributes
        ----------
    """

    def __init__(self, abuns, images, sim_cube_half_length=64,
                 start_params=np.array([1000., 100., 50., 5., 12.])):

        self.abuns = abuns

        # get dict of lines
        self.lines = {}
        for image_id in images.keys():
            if image_id[0] in self.lines:
                self.lines[image_id[0]].append(image_id[1])
            else:
                self.lines[image_id[0]] = [image_id[1]]

        # sort lines
        for key in self.lines.keys():
            self.lines[key].sort()

        # Check we have abundances
        for species in self.lines.keys():
            if species not in abuns:
                raise AttributeError("No {0} abundance supplied".format(
                                        species))

        # Set images for all
        self.images = images

        # Set initial params
        self.params = start_params
        self.param_names = ['distance', 'mean_density', 'depth', 'cfac', 'Tg']

        self.sim_cube_half_length = sim_cube_half_length

    def rademacher_propose(self):
        """ rademacher_propose()

            Propose a random perturbation vector by drawing
            from the Rademacher distribution

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        proposal = np.random.choice([-1, +1], size=5)
        return proposal

    def dist(self, cloud):
        """ chi2(cloud)

            Calculate a chi2 distance between the simulated cloud and
            the observations.

            Paramerets
            ----------
            cloud : LRFCloud
                The simulated cloud that we wish to compare to the
                observations.

            Returns
            -------
            chi2_value : float
                The chi2 combined across images
        """
        dist_value = 0.

        for image_id in self.images.keys():
            comp_image = cloud.image(image_id[0], image_id[1]).flatten()

            dist_value += ks_2samp(self.images[image_id].flatten(),
                                   comp_image)[0]

        return dist_value

    def fit(self, max_iter=100, c_index=1./3., a=1, c=1, samples=1,
            lower_bounds=np.array([10., 1., 1., 1.1, 2.8]),
            upper_bounds=np.array([20000, 1000., 1000., np.inf, np.inf]),
            verbose=False):
        """ fit(max_iter=100, c_index=1./3, a=1, c=1.,,
            lower_bounds=np.array([10., 1., 1., 1.1, 2.8]),
            upper_bounds=np.array([20000, 1000., 1000., np.inf, np.inf])
            verbose=False)

            Perform the SPSA fit.

            We assume that:

            a_n = a/n
            c_n = c/n^c_index

            Where we must have 1/6 < index < 1/2 for convergence

            Parameters
            ----------
            max_iter : int, optional
                The maximum number of iterations for which the fit
                is run
            c_index : float, optional
                The index used
            a : float, optional
                The value of a to use
            c : float, optional
                The value of c to use
            lower_bounds : ndarray, optional
                Lower limits on the parameters
            upper_bound : ndarray, optional
                Upper limits on teh parameters
            verbose : bool, optional
                Controls whether info gets dumped to stdout

            Returns
            -------
            None
        """

        n = 1
        while n <= max_iter:
            perturb = self.rademacher_propose()

            plus_params = self.params + c/pow(n, c_index) * perturb
            minus_params = self.params - c/pow(n, c_index) * perturb

            # keep in bounds
            plus_params = np.minimum(plus_params, upper_bounds)
            plus_params = np.maximum(plus_params, lower_bounds)

            minus_params = np.minimum(minus_params, upper_bounds)
            minus_params = np.maximum(minus_params, lower_bounds)

            plus_dist = 0.
            minus_dist = 0.
            for s in range(samples):
                plus_cloud = LRFCloud(self.sim_cube_half_length,
                                      plus_params[0], plus_params[1],
                                      plus_params[2], plus_params[3], outer_L,
                                      plus_params[4], self.lines, self.abuns)
                minus_cloud = LRFCloud(self.sim_cube_half_length,
                                       minus_params[0], minus_params[1],
                                       minus_params[2], minus_params[3],
                                       outer_L, minus_params[4],
                                       self.lines, self.abuns)

                plus_dist += self.dist(plus_cloud)
                minus_dist += self.dist(minus_cloud)

            plus_dist /= samples
            minus_dist /= samples

            # Note that, as perturbation is drawn from Rademacher dist,
            # 1/Delta_n_i = Delta_n_i
            self.params -= (a/n * (plus_dist - minus_dist) * perturb
                            / (2 * c/pow(n, c_index)))

            # Keep in bounds
            self.params = np.minimum(self.params, upper_bounds)
            self.params = np.maximum(self.params, lower_bounds)

            n += 1

            if verbose:
                print(n, self.params, plus_dist, minus_dist)
