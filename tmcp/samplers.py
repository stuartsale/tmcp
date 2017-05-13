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
                                        prev_cloud.data_list[line_id].shape[0],
                                        prev_cloud.data_list[line_id].shape[1],
                                        prev_cloud.nz)

    # Make first draw on ellipse
    new_angle = 2 * math.pi * np.random.rand()

    # Define angle-bracket
    min_angle = new_angle - 2*math.pi
    max_angle = new_angle

    # create new CloudProbObj
    new_cloud = CloudProbObj.copy_changed_z(prev_cloud, new_zs)

    # Test prob
    if ((new_cloud.log_posterior_prob - prev_clous.log_posteriorprob)
            > log_slice_level):
        return new_cloud

    # Iterate until acceptance
    accepted = False
    while not accepted:

        # Draw new angle
        new_angle = (max_angle - min_angle) * np.random.rand() + min_angle

        # Get new zs
        prop_zs = (math.cos(new_angle) * prev_cloud.zs
                   + math.sin(new_angle) * new_zs)

        # create new CloudProbObj
        new_cloud = CloudProbObj.copy_changed_z(prev_cloud, prop_zs)

        # Test prob
        if ((new_cloud.log_posterior_prob - prev_clous.log_posteriorprob)
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


def update_hypers_MH(prev_cloud, ):
    """ update_hypers_MH(prev_cloud, )

        Sample a new CloudProbObj, keeping the zs held constant,
        but updating hyperparams using a Metropolis-Hastings sampler.

        Parameters
        ----------
        prev_cloud : CloudProbObj
            The previous sampled CloudProbObj
        ****

        Returns
        -------
        new_cloud : CloudProbObj
            The new sampled CloudProbObj
    """
