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

import copy as cp
import math
import numpy as np

from cloud import CloudProbObj


def update_zs_ESS(prev_cloud):
    """ update_zs_ESS(prev_cloud)

        Sample a new CloudProbObj, keeping the hyperparams held
        constant, but updating zs using an elliptical slice sampler.

        Parameters
        ----------
        prev_cloud : CloudProbObj
            The previous sampled CloudProbObj

        Returns
        -------
        new_cloud : CloudProbObj
            The new sampled CloudProbObj
    """
    # define slice level
    slice_level = np.random.rand()
    log_slice_level = math.log(slice_level)

    # Draw some zs to define ellipse
    new_zs = {}
    for line_id in prev_cloud.lines:
        new_zs[line_id] = np.random.randn(
                                        prev_cloud.data_dict[line_id].shape[0],
                                        prev_cloud.data_dict[line_id].shape[1],
                                        prev_cloud.nz)

    # Make first draw on ellipse
    new_angle = 2 * math.pi * np.random.rand()

    # Define angle-bracket
    min_angle = new_angle - 2*math.pi
    max_angle = new_angle

    # create new CloudProbObj
    new_cloud = CloudProbObj.copy_changed_z(prev_cloud, new_zs)

    # Test prob
    if ((new_cloud.log_posteriorprob - prev_cloud.log_posteriorprob)
            > log_slice_level):
        return new_cloud

    # Iterate until acceptance
    accepted = False
    while not accepted:

        # Draw new angle
        new_angle = (max_angle - min_angle) * np.random.rand() + min_angle

        if abs(new_angle) <= 1E-2:
            new_angle = 0.

        # Get new zs
        prop_zs = {}
        for line_id in prev_cloud.zs:
            prop_zs[line_id] = (math.cos(new_angle) * prev_cloud.zs[line_id]
                                + math.sin(new_angle) * new_zs[line_id])

        # create new CloudProbObj
        new_cloud = CloudProbObj.copy_changed_z(prev_cloud, prop_zs)

        # Test prob
        if ((new_cloud.log_posteriorprob - prev_cloud.log_posteriorprob)
                > log_slice_level):
            accepted = True

        # redefine bracket
        else:
            if new_angle <= 0:
                min_angle = new_angle
            else:
                max_angle = new_angle

    # Return new state
    return new_cloud


def update_hypers_MH(prev_cloud, density_prop, ps_prop, inducing_prop,
                     abundances_prop):
    """ update_hypers_MH(prev_cloud, density_prop, ps_prop, inducing_prop,
                         abundances_prop))

        Sample a new CloudProbObj, keeping the zs held constant,
        but updating hyperparams using a Metropolis-Hastings sampler.

        Parameters
        ----------
        prev_cloud : CloudProbObj
            The previous sampled CloudProbObj
        density_prop : dict
        ps_prop : dict
        inducing_prop : dict
        abundances_prop : dict

        Returns
        -------
        new_cloud : CloudProbObj
            The new sampled CloudProbObj
    """
    # Draw new hyperparams

    new_density_func = prev_cloud.density_func.MH_propose(density_prop)
    new_power_spec = prev_cloud.power_spec.MH_propose(ps_prop)
    new_inducing_obj = cp.deepcopy(prev_cloud.inducing_obj)
    new_abundances = cp.deepcopy(prev_cloud.abundances_dict)

    # create new CloudProbObj
    new_cloud = CloudProbObj.copy_changed_hypers(prev_cloud, new_density_func,
                                                 new_power_spec,
                                                 new_inducing_obj,
                                                 new_abundances)

    if (new_cloud.log_posteriorprob - prev_cloud.log_posteriorprob) > 0:
        return new_cloud, True
    elif ((new_cloud.log_posteriorprob - prev_cloud.log_posteriorprob)
            > math.log(np.random.rand())):
        return new_cloud, True
    else:
        return prev_cloud, False
