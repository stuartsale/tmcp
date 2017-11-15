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

import abc
import copy
import numpy as np

import cogs
import cloud
import density
import samplers


class ApmEssMh(object):
    """ class ApmEssMh

        A class for sampling the hyperparams of a cloud using an
        Auxiliary Pseudo-Marginal sampler, following Murray & Graham
        (2016), arxiv:1510.02958

        Attributes
        ----------
        iterations : int
            The number of MCMC iterations to be performed
        burnin : int
            The length of the burn in, expressed in the number of
            iterations
        thin : int
            Controls the thinning of the MCMC chain. Values are only
            stored every thin iterations.
        last_cloud : CloudProbObj
            The most recent CloudProbObj
        mh_prop : dict
            The size of the (Gaussian) Metropolis proposal distributions
        hyper_chain : list
            The MCMC chain for the hyperparameters
        run : bool
            Indicates if MCMC has been run
    """

    def __init__(self, data_dict_files, mh_prop, density_func,
                 power_spec, abundances_dict, inducing_x=None, inducing_y=None,
                 dist_array=None, data_dict_arrays=None,
                 iterations=10000, burnin=5000,
                 thin=10):
        """ __init__(data_dict_in, iterations=10000, mh_prop)

            Initialise a ApmEssMh object

            Parameters
            ----------
            data_dict_in: dict

            iterations : int
                The number of iterations for which the sampler runs
            mh_prop : dict
                For each hyperparam a value giving the width (sd) of
                the Gaussian proposal distribution.

            Returns
            -------
            None
        """
        self.iterations = iterations
        self.thin = thin
        self.burnin = burnin

        self.mh_prop = mh_prop

        self.run = False

        # construct the data dict

        data_dict = {}
        for line_id, files in data_dict_in.items():
            data_dict[line_id] = CloudDataObj.from_fits(line_id[0], line_id[1],
                                                        files[0], files[1])
        if data_dict_arrays is not None:
            for line_id, arrays in data_dict_arrays.items():
                data_dict[line_id] = CloudDataObj(line_id[0], line_id[1],
                                                  arrays[0], arrays[1],
                                                  arrays[2], arrays[3])

        # Store density function, etc
        self.density_func = density_func
        self.power_spec = power_spec
        self.abundances_dict = abundances_dict

        # construct the inducing_obj

        if inducing_x is None or inducing_y is None:
            inducing_obj = cloud.CloudInducingObj.new_from_grid(3, 3, xx, xx)
        else:
            inducing_obj = cloud.CloudInducingObj(inducing_x, inducing_y,
                                                  np.zeros(inducing_x.shape))

        # Set distance array

        if dist_array is None:
            self.dist_array = np.arange(100., 5000., 50)

        # Create CloudProbObj

        self.last_cloud = cloud.CloudProb.new_cloud(density_func, power_spec,
                                                    inducing_obj,
                                                    abundances_dict, data_dict,
                                                    dist_array)

        # set up recarray to store chain

        self.chain_cols = (self.last_cloud.density_func.param_names
                           + self.last_cloud.power_spec.param_names
                           + self.last_cloud.abundances_dict.keys())

        for i in range(self.last_cloud.inducing_obj.nu):
            self.chain_cols.append("u{0:d}".format(i))

        self.hyper_chain = np.rec.fromarrays(np.zeros([len(self.chain_cols),
                                                       floor((self.iterations
                                                              - self.burnin)
                                                             / self.thin)]),
                                             names=self.chain_cols)

    def iterate(self):
        """ iterate()

        """
        for i in range(self.iterations):

            # update zs
            self.last_cloud = samplers.update_zs_ESS(self.last_cloud)

            # update hypers
            self.last_cloud = samplers.update_hypers_MH(
                    self.last_cloud, density_prop, ps_prop, inducing_prop,
                    abundances_prop)

            # Add to chains
            if i > self.burnin and i % thin == 0:
                self.store_to_chain((i-self.burnin)/self.thin)

        self.run = True

    def store_to_chain(self, row):
        """ store_to_chain()

            Parameters
            ----------
            row : int
                The row in the chain to write to

            Returns
            -------
            None
        """
        density_params = self.last_cloud.density_func.param_dict()
        for field in density_params:
            self.hyper_chain[row][field] = density_params[field]

        ps_params = self.last_cloud.power_spec.param_dict()
        for field in ps_params:
            self.hyper_chain[row][field] = ps_params[field]

        for field in self.last_cloud.abundance_dict:
            self.hyper_chain[row][field] = (
                                        self.last_cloud.abundance_dict[field])

        for n in range(self.last_cloud.inducing_obj.nu):
            self.hyper_chain[row]["u{0:d}".format(n)] = (
                    self.last_cloud.inducing_obj.inducing_values[n])

    def mean_sds(self):
        """ mean_sds()

            Calculate the means and sds of the chains

            Parameters
            ----------
            None

            Returns
            -------
            None
        """
        means = {}
        sds = {}

        for field in self.hyper_chain.dtype.names:
            means[field] = np.mean(self.hyper_chain[field])
            sds[field] = np.std(self.hyper_chain[field])
