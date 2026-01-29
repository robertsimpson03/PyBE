#!/usr/bin/env python3
import numpy as np

# This version was prettier but ultimately slower
#def get_field(x, y, a, b):
#    field_inside = 4*(x/a + 1j*y/b)/(a + b)  # Inside the ellipse
#
#    z_conjugate = x-1j*y
#    radical = (z_conjugate)*np.sqrt(1 - (a**2-b**2)/(z_conjugate)**2)
#    field_outside = 4/(z_conjugate + radical)  # Outisde the ellipse
#
#    s_squared = np.sqrt((x/a)**2 + (y/b)**2)  # Ellipse parameter
#    field = np.where(s_squared < 1, field_inside, field_outside)
#
#    return field.real, field.imag

def get_field(x, y, a, b):
    """
        Evaluate the E field from a uniform elliptical distribution.

        Evaluate the electric field at a position or for an araay from a 2D
        Uniform elliptical charge distribution using an analytical formula.

        Parameters
        ----------
        x : float or array_like
            x-coordinates to evaluate field at.
        y : float or array_like
            y-coordinates to evaluate field at.
        a : float
            Ellipse x-radius.
        b : float
            Ellipse y-radius.

        Returns
        -------
        Ex : ndarray
            x-component of the electric field.
        Ey : ndarray
            y-component of the electric field.

    """

    E_x = np.zeros_like(x, dtype=np.float64)
    E_y = np.zeros_like(y, dtype=np.float64)

    x_sqrd, y_sqrd = x**2, y**2
    a_sqrd, b_sqrd = a**2, b**2
    inside = (x_sqrd / a_sqrd + y_sqrd / b_sqrd) <= 1
    out = ~inside

    denom_in = a + b
    E_x[inside] = x[inside] / (a * denom_in)
    E_y[inside] = y[inside] / (b * denom_in)

    x_out, y_out = x[out], y[out]
    x_out_sqrd, y_out_sqrd = x_sqrd[out], y_sqrd[out]

    b = a_sqrd + b_sqrd - x_out_sqrd - y_out_sqrd
    c = a_sqrd*b_sqrd - x_out_sqrd*b_sqrd - y_out_sqrd*a_sqrd
    root = (-b + np.sqrt(b**2 - 4*c)) / 2

    denom_out = root + np.sqrt((a_sqrd+root)*(b_sqrd + root))

    E_x[out] = x_out / (a_sqrd + denom_out)
    E_y[out] = y_out / (b_sqrd + denom_out)
    return E_x, E_y
