import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import InterpolatedUnivariateSpline

from cogs import CoGsObj
from density import QuadraticDensityFunc
from powerspec import SM14_powerspec


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

        if self.data_array.shape != self.error_array.shape:
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
        inducing_values : dict
        inducing_cov_mats : dict
        inducing_cov_mats_cho : dict
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

        self.inducing_values = {}
        self.inducing_cov_mats = {}
        self.inducing_cov_mats_cho = {}

    def add_values(self, line_id, values):
        """ add_values(line_id, values)

            Add the values of the (column density) field at the inducing
            points for some line.

            Parameters
            ----------
            line_id : list
                The species (str) and line(int) of the image concerned
            values : ndarray
                The values of the (column density field at the inducing
                points.
                Must have the same size as the number of inducing points.

            Returns
            -------
            None
        """
        if self.values_check(values):
            self.inducing_values[line_id] = values
        else:
            raise ValueError("The number of values supplied does not match"
                             " the number of inducing points.")

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

    def single_diff(self, x_pos, y_pos):
        """ single_diff(x_pos, y_pos)

            Get the angular differences between a single position
            as supplied and the inducing points.

            Parameters
            ----------
            x_pos : float
                The x position
            y_pos :float
                The y position

            Returns
            -------
            ndarray
                The angular differences between the supplied position
                and the inducing points.
                The shape of this matches that of inducing_x and
                inducing_y.
        """
        diff = np.sqrt(np.power(x_pos - self.inducing_x, 2)
                       + np.power(y_pos - self.inducing_y, 2))
        return diff


class CloudProbObj(object):
    """ This class holds ...

        Attributes
        ----------
        power_spec :
        density_func :
        power_spec :
        inducing_obj : CloudInducingObj
        abundances_dict : dict
        data_dict : list(CloudDataObj)
        dist_array : ndarray
        nz : int
        cogs : CoGsObj

        lines : list
        log_posteriorprob : float
        log_priorprob : float
        log_inducingprob : float
        log_likelihood: float
        mean_dict : dict
        cov_funcs : dict
        inducing_cov_mats : dict
        inducing_cov_mats_cho : dict
    """

    def __init__(self, density_func, power_spec, inducing_obj,
                 abundances_dict, data_dict, dist_array, nz=10):

        self.density_func = density_func
        self.power_spec = power_spec
        self.inducing_obj = inducing_obj
        self.abundances_dict = abundances_dict

        self.data_dict = data_dict

        self.dist_array = dist_array
        self.nz = nz

        self.lines = []

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

        # initialise cogs
        self.cogs = None

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
                self.inducing_cov_mats_cho[line_id] = cho_factor(
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

            Q = cho_solve(self.inducing_cov_mats_cho[line_id],
                          self.inducing_values[line_id]
                          - self.mean_dict[line_id])

        # Combine across lines to get total prob
        # - Implement inter-line covariances!?!

            self.log_inducingprob += (
                - np.sum(np.log(np.diag(
                                    self.inducing_cov_mats_cho[line_id][0])))
                - np.dot(self.inducing_dict[line_id]
                         - self.mean_dict[line_id], Q)/2.)

    def set_zs(self, zs=None):
        """ set_zs()

            Set the random numbers employed in the Monte Carlo
            marginalisation of the unknown column densities for all the
            observed pixels that occurs when calculating the estimated
            (partly marginalised) loglikelihood.

            Parameters
            ----------
            zs : dict or None
                The z values to be stored. If zs is None then new values are
                sampled from the standard normal distribution.

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
        """
        for line_id in self.lines:
            cov_marg = self.cov_funcs[line_id](0)
            for indices in np.ndindex(self.data_dict[lineid].shape):

                # work out mean and sd
                diff = self.inducing_obj.single_diff(
                            self.data_dict[lineid].x_coord[indices],
                            self.data_dict[lineid].x_coord[indices])
                covar_vec = self.cov_funcs[line_id](diff)

                Q = cho_solve(self.inducing_cov_mats_cho[line_id], covar_vec)

                mean_cond = (self.mean_dict[line_id]
                             + np.dot(Q,
                                      self.inducing_values[line_id]
                                      - self.mean_dict[line_id]))
                cov_cond = cov_marg - np.dot(Q, covar_vec)

                # use z and mean and sd to get col dens
                col_dens = mean_cond + cov_cond * self.zs[line_id][indices]

                # use CoG to convert to brightness temp
                TBs = self.cogs(line_id[0], line_id[1], np.log(col_dens))

                # get likelihood
                self.log_likelihood += 1./self.nz * (
                    - np.log(self.data_dict[line_id].error_array[indices])/2.
                    - np.sum(np.power((
                        TBs - self.data_dict[line_id].data_array[indices])
                        / self.data_dict[line_id].error_array[indices], 2))/2.)

    @classmethod
    def new_cloud(cls, density_func, power_spec, inducing_obj, abundances_dict,
                  data_dict, dist_array, nz=10):
        """ new_cloud(density_func, power_spec, inducing_obj, 
                      abundances_dict, data_dict, dist_array, nz=10)

            A factory function to build a new CloudProbObj instance,
            including initially setting probabilities, covariance 
            matrices, etc.

            Parameters
            ----------
            density_func : DensityFunc or derived
            power_spec : IsmPowerspec or derived
            inducing_obj : CloudInducingObj
            abundances_dict : dict
            data_dict : dict
            dist_array : ndarray
            nz : int

            Returns
            -------
            CloudProbObj
                The new instance with probabilities etc all set.
        """
        new_obj = cls(density_func, power_spec, inducing_obj, abundances_dict,
                      data_dict, dist_array, nz)

        for image in new_obj.data_list:
            new_obj.lines.append([image.species, image.line])

        # Cycle through observed lines
        for line_id in new_obj.lines:

            crit_dens = critical_density(line_id[0], line_id[1])

            # Trim mean density function
            censored_dens = new_obj.density_func.censored_grid(
                                                new_obj.dist_array, crit_dens)

            # Get mean column density in >0 region
            new_obj.mean_dict[line_id] = np.sum(censored_dens)

            # Project powerspectrum
            ps = new_obj.power_spec.project(new_obj.dist_array, censored_dens)

            # Obtain covariance function
            cov_values = np.fft.irfft(ps)
            new_obj.cov_funcs[line_id] = InterpolatedUnivariateSpline(
                                                new_obj.dist_array, cov_values)

        # Get CoGs
        line_dict = {}
        for line_id in new_obj.lines:
            if line_id[0] in line_dict:
                line_dict[line_id[0]].append(line_id[1])
            else:
                line_dict[line_id[0]] = [line_dict[1]]

        nHmean = new_obj.density_func.limited_mean()

        new_obj.cogs = CoGsObj(self.abundance_dict, line_dict, nHmean)

        # Estimate probs

        new_obj.set_prior_logprob()
        new_obj.set_inducing_cov_matrices()
        new_obj.set_inducing_logprob()
        new_obj.set_zs()
        new_obj.estimate_loglikelihood()

        new_obj.log_posteriorprob = (new_obj.log_priorprob
                                     + new_obj.log_inducingprob
                                     + new_obj.log_likelihood)

        return new_obj

    @classmethod
    def copy_changed_z(cls, prev_cloud, zs):
        """ copy_changed_z(prev_cloud, zs)

            A factory method to produce a new CloudProbObj instance
            that copies all hyperparams (i.e. mean_func, power_spec,
            inducing_obj, abundances) from a previous instance, but
            the new instance has newly supplied zs.

            Parameters
            ----------
            prev_cloud : CloudProbObj
                The previous instance, upon which the new instance
                is largely based.
            zs : dict(ndarray)
                A dict containing ndarrays of new zs for each line_id

            Returns
            -------
            CloudProbObj
                The new instance with the hyperparams of the previous
                instance and new zs
        """
        new_obj = cls(prev_cloud.density_func, prev_cloud.power_spec,
                      prev_cloud.inducing_obj, prev_cloud.abundances_dict,
                      prev_cloud.data_dict, prev_cloud.dist_array,
                      prev_cloud.nz)

        new_obj.lines = prev_cloud.lines

        new_obj.mean_dict = prev_cloud.mean_dict
        new_obj.cov_funcs = prev_cloud.cov_funcs

        new_obj.cogs = prev_cloud.cogs

        new_obj.log_priorprob = prev_cloud.log_priorprob
        new_obj.log_inducingprob = prev_cloud.log_inducingprob

        new_obj.inducing_cov_mats = prev_cloud.inducing_cov_mats
        new_obj.inducing_cov_mats_cho = prev_cloud.inducing_cov_mats_cho

        # get new likelihood
        new_obj.set_zs(zs)
        new_obj.estimate_loglikelihood()

        new_obj.log_posteriorprob = (new_obj.log_priorprob
                                     + new_obj.log_inducingprob
                                     + new_obj.log_likelihood)

        return new_obj


def critical_density(species, line):
    """ critical_density(species, line)

        Find the critical density of a given transition line

        Parameters
        ----------
        species : str
        line : int

        Returns
        -------
        float
            The critical density
    """
    return 0.
