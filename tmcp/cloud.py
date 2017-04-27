import numpy as np
import scipy.linalg as slg

from powerspec import SM14_powerspec
from cogs import CoGsObj

class CloudDataObj(object):
    """ A class to hold an image for some band

        Attributes
        ----------
        species : string
        line : int
        x_coord : ndarray
        y_coord : ndarray
        data_array : ndarray
        error_array : ndarray
        shape : ndarray
    """
    def __init__(self, species, line, x_coord, y_coord, data_array,
                 error_array):
        self.species = species
        self.line = line
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.data_array = data_array
        self.error_array = error_array

        if self.data_array.shape != self.error_array.shape
            raise ValueError("Data and error images are not the same shape")
        if len(self.data_array.shape) != 2:
            raise ValueError("Data image is not 2d")
        self.shape = self.data_array.shape


class CloudInducingObj(object):
    """

        Attributes
        ----------
        inducing_x : ndarray
        inducing_y : ndarray
        nu : int
        inducing_diff : ndarray
    """

    def __init__(self, inducing_x, inducing_y, inducing_vals):
        # make sure both sets of positions are 1D ndarrays
        # get inducing_diff 2D (matrix) ndarray

        self.inducing_x = np.array(inducing_x).flatten()
        self.inducing_y = np.array(inducing_y).flatten()

        if self.inducing_x.size != self.inducing_y.size:
            raise ValueError("Dimensions of x & y inducing point arrays "
                             "do not match")

        self.nu = self.inducing_x.size
        self.inducing_diff = np.sqrt(np.power(self.inducing_x
                                              - self.inducing_x.reshape(
                                                    self.nu, 1), 2)
                                     + np.power(self.inducing_y
                                                - self.inducing_y.reshape(
                                                    self.nu, 1), 2))

    def values_check(self, values):
        """ values_check(values)

            Verify that the shape of an array of inducing values matches
            the shape of the x&y coordinates of the inducing points.

            Parameters
            ----------
            values : ndarray
                The array of values at the inducing points

            Returns
            -------
            bool
            Whether the shapes match
        """
        return values.shape == self.inducing_x.shape


class CloudProbObj(object):
    """ This class holds ...

        Attributes
        ----------
        power_spec : 
        inducing_dict : dict
        abundances_dict : dict
        data_list : list
        nz : int
        lines : dict
        log_posteriorprob : float
        means : dict
        inducing_cov_mat : dict
    """

    def __init__(self, density_func, power_spec, inducing_dict,
                 abundances_dict, data_list, inducing_obj, nz=10):
        """ __init__()
        """

        self.density_func = density_func
        self.power_spec = power_spec
        self.inducing_dict = inducing_dict
        self.abundances_dict = abundances_dict
        self.data_list = data_list
        self.inducing_obj = inducing_obj
        self.nz = nz

        self.lines = []
        for image in self.data_list:
            self.lines.append([image.species, image.line])

        # initialise (log) probs to 0
        self.log_posteriorprob = 0.
        self.log_priorprob = 0.
        self.log_inducingprob = 0.
        self.log_likelihood = 0.

        # initialise empty dicts to hold means & cov mats
        self.mean_dict = {}
        self.cov_funcs = {}
        self.inducing_cov_mats = {}
        self.inducing_cov_mats_cho = {}

        # Obtain density function
        # Cycle through observed lines
        for line_id in self.lines:
            # Check shape of inducing valyues OK
            self.inducing_obj.values_check(self.inducing_dict[line_id])

            # Trim mean density function
            # - set to 0 any density < critical density
            # Get mean column density in >0 region
            # Project powerspectrum
            # Obtain covariance function
            self.cov_funcs[line_id] = 


        # Get CoGs

    def __getattr__(self, attr):
        """ __getattr__(attr)

            Grab attributes from inducing_obj used in composition of
            CloudProbObj.
        """
        return getattr(self.inducing_obj, attr)

    def set_inducing_cov_matrices(self):
        """ set_inducing_cov_matrices()

            Set the covariance matrices for the values at the inducing
            points.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        # Cycle through lines
        for line_id in self.lines:

        # Fill in cov matrices
            self.inducing_cov_mats[line_id] = self.cov_funcs[line_id](
                                                    self.inducing_diff)

        # Get cholesky decompositions
            try:
                self.inducing_cov_mats_cho[line_id] = slg.cho_factor(
                                               self.inducing_cov_mats[line_id],
                                               check_finite=False)
            except np.linalg.linalg.LinAlgError or ValueError:
                raise

    def set_prior_logprob(self):
        """ set_prior_logprob()

            Calculate the log-(hyper) prior probability.
            Currently Assumes a uniform prior.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        # Uniform prior
        self.log_priorprob = 0.

    def set_inducing_logprob(self):
        """ set_inducing_logprob()

            Calculate the log-probability of the values at the inducing
            points, i.e. $\pr(\vu | \hyper)$.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        # Get cov matrices
        self.set_inducing_cov_matrices()

        # Cycle through lines
        for line_id in self.lines:

            # calculate prob

            Q = slg.cho_solve(self.inducing_cov_mats_cho[line_id],
                              self.inducing_dict[line_id]
                              - self.mean_dict[line_id])

        # Combine across lines to get total prob
        # - Implement inter-line covariances!?!

            self.log_inducingprob += (
                - np.sum(np.log(np.diag(
                                    self.inducing_cov_mats_cho[line_id][0])))
                - np.dot(self.inducing_dict[line_id]
                         - self.mean_dict[line_id], Q)/2.)

    def set_zs(self):
        """ set_zs()

            Set the random numbers employed in the Monte Carlo
            marginalisation of the unknown column densities for all the
            observed pixels that occurs when calculating the estimated
            (partly marginalised) loglikelihood.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        self.zs = {}
        for line_id in self.lines:
            self.zs[line_id] = np.random.randn(*self.data_list[line_id].shape,
                                               self.nz)

    def estimate_loglikelihood(self):
        """ estimate_loglikelihood()

            Estimate the log-likelihood.
            Performs a Monte-Carlo estimate of the true log-likelihood,
            marginalising over the column density in each pixel.

            Parameters
            ----------
            None

            Returns
            -------
            None
