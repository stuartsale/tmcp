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
import numpy as np


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

    @classmethod
    def from_fits(cls, species, line, data_file, error_file):
        """ from_fits(species, line, data_file, error_file)

            Construct a CloudDataObj instanceusing fits files as inputs.

            Parameters
            ----------
            species : str
                The species of the line
            line : int
                The energy level of the line
            data_file : str
                The filename of the data file
            error_file : str
                The filename of the error file

            Returns
            -------
            CloudDataObj
                The instance produced from the fits files
        """
        data_hdu = fits.open(data_file)
        data_array = data_hdu[-1].data

        error_hdu = fits.open(error_file)
        error_array = error_hdu[-1].data

        if data_array.shape != error_array.shape:
            raise IndexError("Shape of data and error arrays do not match")

        x = np.arange(data_array.shape[0])
        y = np.arange(data_array.shape[1])
        X, Y = np.meshgrid(x, y)

        data_wcs = wcs.WCS(data_hdu[-1].header)

        data_x, data_y = data_wcs.wcs_pix2world(X, Y, 0)

        return cls.__init__(species, line, data_x, data_y, data_array,
                            error_array)

    def onsky_limits(self):
        """ onsky_limits()

            Return the minimum and maximum coordinates on each
            axis - typically (RA, DEC) or (l, b).

            Parameters
            ----------
            None

            Returns
            -------
            (x_min, x_max) : tuple
                The min and max coordinates along the first axis
                (typically RA or l)
            (y_min, y_max) : tuple
                The min and max coordinates along the second axis
                (typically DEC or b)
        """
        x_lims = (self.x_coord.min(), self.x_coord.max())
        y_lims = (self.y_coord.min(), self.y_coord.max())

        return x_lims, y_lims
