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


class CloudDataSet(object):
    """ A class to hold a dict of images for several lines/bands

        Paramaters
        ----------
        files_dict : dict
        arrays_dict : dict, optional

        Attributes
        ----------
        data_objs : dict
        max_opening_angle : float
    """

    def __init__(self, files_dict, arrays_dict=None):

        self.data_objs = {}
        for line_id, files in files_dict.items():
            self.data_objs[line_id] = CloudDataObj.from_fits(line_id[0],
                                                             line_id[1],
                                                             files[0],
                                                             files[1])
        if arrays_dict is not None:
            for line_id, arrays in arrays_dict.items():
                self.data_objs[line_id] = CloudDataObj(line_id[0], line_id[1],
                                                       arrays[0], arrays[1],
                                                       arrays[2], arrays[3])

        lims = self.onsky_limits()
        self.max_opening_angle = 1.5 * max(abs(lims[0][1] - lims[0][0]),
                                           abs(lims[1][1] - lims[1][0]))

    def __getattr__(self, attr):
        return self.data[attr]

    def __iter__(self):
        for data_obj in self.data_objs.values():
            yield data_obj

    def onsky_limits(self):
        """ onsky_limits()

            Return the minimum and maximum coordinates from any of the
            images. (Min, Max) pairs are provided on each axis
            - typically (RA, DEC) or (l, b).

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
        x_min = +np.inf
        x_max = -np.inf
        y_min = +np.inf
        y_max = -np.inf

        for image in self.data_objs.values():
            lims = image.onsky_limits()

            x_min = min(x_min, lims[0][0])
            x_max = max(x_max, lims[0][1])
            y_min = min(y_min, lims[1][0])
            y_max = max(y_max, lims[1][1])

        return ((x_min, x_max), (y_min, y_max))
