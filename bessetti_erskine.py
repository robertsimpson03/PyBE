import numpy as np
from scipy.special import wofz

def _bassetti_erskine(x, y, sigma_x, sigma_y):
    y_neg = -np.abs(y) # Compute for lower region

    z_conjugate = x - 1j*y_neg
    omega_conjugate = x*sigma_y/sigma_x - 1j*y_neg*sigma_x/sigma_y

    xi_sqrd = (x/sigma_x)**2+(y_neg/sigma_y)**2
    sigma_sqrd_diff = sigma_x**2-sigma_y**2
    s1 = z_conjugate/np.sqrt(2*sigma_sqrd_diff)
    s2 = omega_conjugate/np.sqrt(2*sigma_sqrd_diff)

    prefactor = 1j*np.sqrt(2*np.pi/sigma_sqrd_diff)
    field_positive = prefactor * (np.exp(-xi_sqrd/2)*wofz(s2) - wofz(s1))
    
    field = np.where(y <= 0, 
                     field_positive, 
                     field_positive.conjugate())
    
    return field


def gaussian_field(x, y, sigma_x, sigma_y, flag_complex_out=False):
    """
        Evaluate the electric field from a 2D Gaussian charge distribution.

        Evaluate the electric field from a 2D Gaussian charge distribution
         using the Bassetti-Erskine formula.

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
        flag_complex_out : bool, optional
            If True, returns complex field (Ex + 1j*Ey).
            If False, returns separate real and imaginary parts (Ex, Ey).
            Default is False.

        Returns
        -------
        field : complex or tuple
            If flag_complex_out is True:
                complex ndarray
                    Complex field values (Ex + 1j*Ey)
            If flag_complex_out is False:
                tuple of (Ex, Ey) where:
                Ex : ndarray
                    Real part of field (x-component)
                Ey : ndarray 
                    Imaginary part of field (y-component)
    """
    if sigma_x == sigma_y:  # circular
        field = 2/(x-1j*y)*(1-np.exp(-(x**2+y**2)/(2*sigma_x**2)))

    elif sigma_x > sigma_y:
        field = _bassetti_erskine(x, y, sigma_x, sigma_y)

    else:  # Calculate on reflected axes
        field = _bassetti_erskine(y, x, sigma_y, sigma_x)
        field = 1j*field.conjugate() # Reflect back to correct axes

    field = np.where((x==0) & (y==0), 0+0j, field) # Avoid `nan` at origin
    
    if flag_complex_out==False:
        return field.real, field.imag
    else:
        return field

def uniform_field(x, y, a, b, flag_complex_out=False):
    """
        Evaluate the E field from a uniform elliptical distribution.

        Evaluate the electric field at a position or for a grid from a 2D
        Uniform elliptical charge distribution using.

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
        flag_complex_out : bool, optional
            If True, returns complex field (Ex + 1j*Ey).
            If False, returns separate real and imaginary parts (Ex, Ey).
            Default is False.

        Returns
        -------
        field : complex or tuple
            If flag_complex_out is True:
                complex ndarray
                    Complex field values (Ex + 1j*Ey)
            If flag_complex_out is False:
                tuple of (Ex, Ey) where:
                Ex : ndarray
                    Real part of field (x-component)
                Ey : ndarray 
                    Imaginary part of field (y-component)
    """

    z_conjugate= x-1j*y
    s = lambda x, y : np.sqrt((x/a)**2 + (y/b)**2) # Ellipse parameter
    
    field_inside = 4*(x/a + 1j*y/b)/(a + b) # Inside the ellipse

    radical = (z_conjugate)*np.sqrt(1 - (a**2-b**2)/(z_conjugate)**2)
    field_outside = 4/(z_conjugate + radical) # Outisde the ellipse

    field = np.where(s(x, y) < 1, field_inside, field_outside)

    if flag_complex_out==False:
        return field.real, field.imag
    else:
        return field