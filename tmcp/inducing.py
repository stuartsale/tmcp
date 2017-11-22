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

        self.x = np.array(inducing_x).flatten()
        self.y = np.array(inducing_y).flatten()

        if self.x.shape != self.y.shape:
            raise ValueError("Dimensions of x & y inducing point arrays "
                             "do not match")

        self.shape = self.x.shape
        self.nu = self.x.size

        self.diff = np.sqrt(np.power(self.x - self.x.reshape(self.nu, 1), 2)
                            + np.power(self.y - self.y.reshape(self.nu, 1), 2))

        self.values = inducing_vals.flatten()
        self.ov_mat = np.zeros([self.nu, self.nu])
        self.cov_mat_cho = np.zeros([self.nu, self.nu])
        self.mean = 0.

    def add_values(self, values):
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
            self.values = values
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
        return values.shape == self.x.shape

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
        diff = np.sqrt(np.power(x_pos - self.x, 2)
                       + np.power(y_pos - self.y, 2))
        return diff

    def multi_diff(self, x_pos, y_pos):
        """ multi_diff(x_pos, y_pos)

            Get the angular differences between a multiple positions
            as supplied and the inducing points.

            Parameters
            ----------
            x_pos : ndarray
                The x positions
            y_pos :ndarray
                The y positions, shape must match that of x_pos

            Returns
            -------
            ndarray
                The angular differences between the supplied position
                and the inducing points.
                The shape of this is (x_pos.size, nu)
        """
        diff = np.sqrt(np.power(x_pos.flatten().reshape(-1, 1)
                                - self.x.flatten(), 2)
                       + np.power(y_pos.flatten().reshape(-1, 1)
                                  - self.y.flatten(), 2))
        return diff

    @classmethod
    def random_sample(cls, prev_obj):
        """ random_sample()

            Radomly draw a new CloudInducingObj which shares most
            of the same attributes as the previous object, but
            with values that are drawn from the (shared) mean
            and covariance matrix.

            Parameters
            ----------
            prev_obj : CloudInducingObj

            Returns
            -------
            new_obj : CloudInducingObj
        """
        new_obj = cp.deepcopy(prev_obj)

        cho_factor = np.linalg.cholesky(new_obj.cov_mat)
        new_obj.inducing_values = (new_obj.mean
                                   + np.dot(cho_factor,
                                            np.random.randn(new_obj.nu)))

        return new_obj

    @classmethod
    def new_from_grid(cls, nx, ny, x_range, y_range):
        """ new_from_grid(nx, ny, x_range, y_range)

            Create a new CloudInducingObj instance that places
            the inducing points on a regular nx x ny grid.

            Parameters
            ----------
            nx : int
                The length of the grid in the x direction (typically
                RA or l)
            ny : int
                The length of the grid in the y direction (typically
                DEC or b)
            x_range : list
                A length 2 list that gives the range of x-values that
                the onservations span, in form [x_min, x_max]
            y_range : list
                A length 2 list that gives the range of y-values that
                the onservations span, in form [y_min, y_max]
        """
        x_step = (x_range[1]-x_range[0]) / nx
        y_step = (y_range[1]-y_range[0]) / ny

        Xs = x_range[0] + x_step * (np.arange(nx) + 0.5)
        Ys = y_range[0] + y_step * (np.arange(ny) + 0.5)

        x_pos, y_pos = np.meshgrid(Xs, Ys)
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()

        return cls(x_pos, y_pos, np.zeros(x_pos.shape))

    @classmethod
    def weighted_add(cls, weight1, inducing1, weight2, inducing2, check=True):
        """ weighted_add(weight1, inducing1, weight2, inducing2
                         check=True)

            Make a weighted combination of two CloudInducingObj
            instances, optionally checking positions, covariance
            matrices, etc agree.

            Parameters
            ----------
            weight1 : float
            inducing1 : CloudInducingObj
            weight2 : float
            inducing2 : CloudInducingObj
            check : bool, optional

            Returns
            -------
            new_inducing_obj : CloudInducingObj
        """
        if check:
            if (np.all(inducing1.inducing_x != inducing2.inducing_x)
                or np.all(inducing1.inducing_y != inducing2.inducing_y)
                or inducing1.nu != inducing2.nu
                or np.all(inducing1.inducing_cov_mat
                          != inducing2.inducing_cov_mat)):
                    raise ValueError("Two inducing point objects are not "
                                     "compatable")

        cho_factor = np.linalg.cholesky(new_obj.cov_mat)
        cho_inv = np.linalg.inv(cho_factor)

        inducing_zs1 = np.dot(cho_inv, (inducing1.values - inducing1.mean))
        inducing_zs2 = np.dot(cho_inv, (inducing2.values - inducing2.mean))

        new_inducing_zs = weight1 * inducing_zs1 + weight2 * inducing_zs2
        new_values = (inducing1.mean + np.dot(cho_factor, new_inducing_zs))

        new_inducing_obj = cp.deepcopy(inducing1)
        new_inducing_obj.values = new_values

        return new_inducing_obj
