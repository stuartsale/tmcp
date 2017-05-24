import copy
import numpy as np

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
    """

    def __init__(self, data_dict_in, mh_prop, density_func=None,
                 power_spec=None, inducing_x=None, inducing_y=None,
                 abundances_dict=None, dist_array=None,
                 iterations=10000, durnin=5000,
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

        self.chain = []

        # construct the data dict

        data_dict = {}
        for line_id, files in data_dict_in.items():
            data_dict[line_id] = CloudDataObj.from_fits(line_id[0], line_id[1],
                                                        files[0], files[1])

        # Construct DensityFunc instance

        if density_func is None:
            density_func = density.QuadraticDensityFunc(10, XX, 2)

        # construct power spectrum

        if power_spec is None:
            power_spec = 

        # construct the inducing_obj

        if inducing_x is None or inducing_y is None:
            inducing_obj = cloud.CloudInducingObj.new_from_grid(3, 3, xx, xx)
        else:
            inducing_obj = cloud.CloudInducingObj(inducing_x, inducing_y,
                                                  np.zeros(inducing_x.shape))

        # Set initial guess at abundances

        if abundances_dict is None:
            pass

        # Set distance array

        if dist_array is None:
            pass

        # Create CloudProbObj

        self.last_cloud = cloud.CloudProb.new_cloud(density_func, power_spec,
                                                    inducing_obj,
                                                    abundances_dict, data_dict,
                                                    dist_array)

    def iterate(self):
        """ iterate()

        """
        for i in range(self.iterations):

            # update zs
            self.last_cloud = samplers.update_zs_ESS(self.last_cloud)

            # update hypers
            self.last_cloud = samplers.update_hypers_MH()

            # Add to chains
            if i > self.burnin and i % thin == 0:
                self.store_to_chain()

    def store_to_chain(self):
        """ store_to_chain()

        """
        pass
