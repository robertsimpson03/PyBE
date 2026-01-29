#!/usr/bin/env python3

import numpy as np
from scipy.special import wofz

TOLERANCE = 1e-14
def get_field(x, y, sigma_x, sigma_y):
    """
        Evaluate the electric field from a 2D Gaussian charge distribution.

        Evaluate the electric field from a 2D Gaussian charge distribution
        using the circular analytical or Bassetti-Erskine semi-analytical
        formula.

        Parameters
        ----------
        x : float or array_like
            x-coordinates to evaluate field at.
        y : float or array_like
            y-coordinates to evaluate field at.
        sigma_x : float
            Gaussian width in the x direction.
        sigma_y : float
            Gaussian width in the y direction.

        Returns
        -------
        Ex : ndarray
            x-component of the electric field.
        Ey : ndarray
            y-component of the electric field.

    """

    if abs(sigma_x - sigma_y) < TOLERANCE:
        E_x, E_y = _circular(x, y, sigma_x)
    else:
        if sigma_x > sigma_y:
            E_x, E_y = _elliptical(x, y, sigma_x, sigma_y)
        else:
            E_y, E_x = _elliptical(y, x, sigma_y, sigma_x)

    return E_x, E_y


def _circular(x, y, sigma):
    r_squared = x**2 + y**2
    charge_enclosed = - np.expm1(-r_squared/sigma**2)

    common = 2 * np.divide(charge_enclosed, r_squared,
                           np.zeros_like(r_squared), where=r_squared != 0)

    return x * common, y * common

def _elliptical(x, y, sigma_x, sigma_y):
    E_x, E_y = _bassetti_erskine(x, abs(y), sigma_x, sigma_y)
    #E_y *= np.where(y <= 0, 1, -1)
    mask = y < 0
    E_y[mask] *= -1
    return E_x, E_y


def _bassetti_erskine(x, y, sigma_x, sigma_y):
    z = x + 1j*y
    omega = x*sigma_y/sigma_x + 1j*y*sigma_x/sigma_y

    xi_sqrd = (x/sigma_x)**2+(y/sigma_y)**2
    denom = 1 / np.sqrt(2 * (sigma_x**2-sigma_y**2))
    s1 = z * denom
    s2 = omega * denom
    prefactor = 2j*np.sqrt(np.pi) * denom

    field = prefactor * (np.exp(-xi_sqrd/2)*wofz(s2) - wofz(s1))

    return field.real, -field.imag
