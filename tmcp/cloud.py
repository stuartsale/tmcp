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

from astropy.io import fits
import copy as cp
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import InterpolatedUnivariateSpline

from cogs import CoGsObj
from density import QuadraticDensityFunc
from inducing import CloudInducingObj
from powerspec import SM14Powerspec


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
        col_mean : float
        cov_func : method
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

        self.lines = self.data_dict.lines

        # initialise (log) probs to 0
        self.log_posteriorprob = 0.
        self.log_priorprob = 0.
        self.log_inducingprob = 0.
        self.log_likelihood = 0.

        # initialise column mean & cov mats
        self.col_mean = None
        self.cov_func = None

        # initialise cogs
        self.cogs = None

        # calculate angular distances between data pixels and inducing points
        self.inducing_pixel_diffs = {}
        for line_id in self.lines:
            self.inducing_pixel_diffs[line_id] = (
                self.inducing_obj.multi_diff(self.data_dict[line_id].x_coord,
                                             self.data_dict[line_id].y_coord))

    def __deepcopy__(self, memo):
        """ ___deeppcopy__(memo)

            A customised deepcopy that avoids the potentially expensive
            (in both memory and CPU) deep copying of data_dict.

            Parameters
            ----------
            memo : dict
                Used to keep track of what attributes have been copied

            Returns
            -------
            CloudProbObj
                A new instance that shares the same data_dict
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k is not "data_dict":
                setattr(result, k, cp.deepcopy(v, memo))
        result.data_dict = self.data_dict
        return result

    def set_inducing_cov_matrix(self):
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
        # mean
        self.inducing_obj.inducing_mean = self.col_mean

        # Fill in cov matrices
        self.inducing_obj.inducing_cov_mat = (
                                self.cov_func(self.inducing_obj.inducing_diff))

        # Get cholesky decompositions
        try:
            self.inducing_obj.inducing_cov_mat_cho = cho_factor(
                                self.inducing_obj.inducing_cov_mat,
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
        self.set_inducing_cov_matrix()

        # calculate prob

        Q = cho_solve(self.inducing_obj.inducing_cov_mat_cho,
                      self.inducing_obj.inducing_values - self.col_mean)

        self.log_inducingprob = (
            - np.sum(np.log(np.diag(
                self.inducing_obj.inducing_cov_mat_cho[0])))
            - np.dot(self.inducing_obj.inducing_values - self.col_mean, Q)/2.)

    def random_zs(self, zs=None):
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
            self.zs[line_id] = np.random.randn(
                                        self.data_dict[line_id].shape[0],
                                        self.data_dict[line_id].shape[1],
                                        self.nz)

    def set_conditional_moments(self):
        """ set_conditional_moments()

            Set the mean and variance for each pixel conditioned
            on given hyperparameters and values at the inducing
            points.

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        self.mean_cond = {}
        self.var_cond = {}
        for line_id in self.lines:
            var_marg = self.cov_func(0.)

            # work out mean and sd
            covar_mat = self.cov_func(self.inducing_pixel_diffs[line_id]).T

            Q = cho_solve(self.inducing_obj.inducing_cov_mat_cho,
                          covar_mat)

            self.mean_cond[line_id] = (
                self.col_mean + np.dot(Q.T, self.inducing_obj.inducing_values
                                       - self.col_mean))
            self.var_cond[line_id] = var_marg - np.sum(Q*covar_mat, axis=0)

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
        self.log_likelihood = 0.
        for line_id in self.lines:

            # use z and mean and sd to get col dens
            col_dens = (self.mean_cond[line_id]
                        + self.var_cond[line_id]
                        * self.zs[line_id].reshape(-1, self.nz).T)

            TBs = self.cogs(line_id[0], line_id[1], np.log(col_dens))

            self.log_likelihood += -np.sum(np.power(
                    (TBs - self.data_dict[line_id].data_array.flatten())
                    / self.data_dict[line_id].error_array.flatten(), 2))/2.

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

        # Get mean column density in >0 region
        new_obj.col_mean = new_obj.density_func.integral()

        # Project powerspectrum
        ks, ps, f2 = new_obj.power_spec.project(new_obj.dist_array,
                                                new_obj.density_func)

        # Obtain covariance function
        cov_values = np.fft.irfft(ps)
        new_obj.cov_func = InterpolatedUnivariateSpline(new_obj.dist_array,
                                                        cov_values)

        # Set inducing values to initially match mean
        new_obj.inducing_obj.inducing_values = (
                        np.zeros(new_obj.inducing_obj.nu) + new_obj.col_mean)

        # Get CoGs
        line_dict = {}
        for line_id in new_obj.lines:
            if line_id[0] in line_dict:
                line_dict[line_id[0]].append(line_id[1])
            else:
                line_dict[line_id[0]] = [line_id[1]]

        new_obj.cogs = CoGsObj(new_obj.abundances_dict, line_dict,
                               new_obj.density_func, power_spec)

        # Estimate probs

        new_obj.set_prior_logprob()
        new_obj.set_inducing_cov_matrix()
        new_obj.set_inducing_logprob()
        new_obj.random_zs()

        new_obj.set_conditional_moments()
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
        # get copy
        new_obj = cp.deepcopy(prev_cloud)

        # get new likelihood
        new_obj.zs = cp.deepcopy(zs)

        new_obj.estimate_loglikelihood()

        new_obj.log_posteriorprob = (new_obj.log_priorprob
                                     + new_obj.log_inducingprob
                                     + new_obj.log_likelihood)

        return new_obj

    @classmethod
    def copy_changed_hypers(cls, prev_cloud, density_func, power_spec,
                            inducing_obj, abundances_dict):
        """ copy_changed_z(prev_cloud, zs)

            A factory method to produce a new CloudProbObj instance
            that copies the zs from a previous instance, but the new
            instance has newly supplied all hyperparams (i.e. mean_func,
            power_spec, inducing_obj, abundances)

            Parameters
            ----------
            prev_cloud : CloudProbObj
                The previous instance, upon which the new instance
                is largely based.
            density_func : DensityFunc or derived
            power_spec : IsmPowerspec or derived
            inducing_obj : CloudInducingObj
            abundances_dict : dict

            Returns
            -------
            CloudProbObj
                The new instance with the zs of the previous
                instance and new hyperparams
        """
        # get copy
        new_obj = cp.deepcopy(prev_cloud)

        # update hyperparams
        new_obj.density_func = density_func
        new_obj.power_spec = power_spec
        new_obj.inducing_obj = inducing_obj
        new_obj.abundances_dict = abundances_dict

        # Get mean column density in >0 region
        new_obj.col_mean = new_obj.density_func.integral()

        # Project powerspectrum
        ks, ps, f2 = new_obj.power_spec.project(new_obj.dist_array,
                                                new_obj.density_func)

        # Obtain covariance function
        cov_values = np.fft.irfft(ps)
        new_obj.cov_func = InterpolatedUnivariateSpline(new_obj.dist_array,
                                                        cov_values)

        # Get CoGs
        line_dict = {}
        for line_id in new_obj.lines:
            if line_id[0] in line_dict:
                line_dict[line_id[0]].append(line_id[1])
            else:
                line_dict[line_id[0]] = [line_id[1]]

        new_obj.cogs = CoGsObj(new_obj.abundances_dict, line_dict,
                               new_obj.density_func, power_spec)

        # Estimate probs

        new_obj.set_prior_logprob()
        new_obj.set_inducing_cov_matrix()
        new_obj.set_inducing_logprob()

        new_obj.set_conditional_moments()
        new_obj.estimate_loglikelihood()

        new_obj.log_posteriorprob = (new_obj.log_priorprob
                                     + new_obj.log_inducingprob
                                     + new_obj.log_likelihood)

        return new_obj

    @classmethod
    def copy_changed_inducing(cls, prev_cloud, inducing_obj):
        """ copy_changed_inducing(prev_cloud, copy_changed_inducing)

            A factory method to produce a new CloudProbObj instance
            that copies the zs and hyperparams from a previous instance,
            but the new instance has newly supplied inducing_obj

            Parameters
            ----------
            prev_cloud : CloudProbObj
                The previous instance, upon which the new instance
                is largely based.
            inducing_obj : CloudInducingObj

            Returns
            -------
            CloudProbObj
                The new instance with the zs of the previous
                instance and new hyperparams
        """
        # get copy
        new_obj = cp.deepcopy(prev_cloud)

        new_obj.inducing_obj = inducing_obj

        # Estimate probs

        new_obj.set_inducing_logprob()

        new_obj.set_conditional_moments()
        new_obj.estimate_loglikelihood()

        new_obj.log_posteriorprob = (new_obj.log_priorprob
                                     + new_obj.log_inducingprob
                                     + new_obj.log_likelihood)

        return new_obj
