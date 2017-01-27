from __future__ import print_function, division
import math
import numpy as np


class ISM_powerspec(object):
    pass


class SM14_powerspec(ISM_powerspec):

    def __init__(self, gamma=11/3, omega=0, L=1.):
        if gamma < 0:
            raise AttributeError("gamma<=0 implies infinite variance!")
        if omega < 0:
            raise AttributeError("omega<=0 implies infinite variance!")
        if L < 0:
            raise AttributeError("Scale length cannot be negative!")

        self.gamma = gamma
        self.omega = omega
        self.L = L

        # Normalisation

        self.R = 1
        k = np.arange(0, self.L*1000, self.L/100)
        unnormalised_ps = self.PS(k)

        self.R = 1/(4*math.pi*np.trapz(unnormalised_ps*np.power(k, 2), k))

    def PS(self, k):

        ps = (self.R * np.power(k*self.L, 2*self.omega)
              / np.power(1 + np.power(k*self.L, 2), self.gamma/2+self.omega))
        return ps
