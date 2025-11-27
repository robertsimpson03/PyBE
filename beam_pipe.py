import numpy as np
from bessetti_erskine import gaussian_field, uniform_field

def beam_pipe_field(x, y, beam_pos_x, beam_pos_y,
                    regular_field, pipe_radius,
                    flag_complex_out=False):
    """
        Evaluate electric field in beam pipe given a know open solution

        Computes the electric field from from a distribution within a beampipe
        given a known open-boundary solution using the method of images charges

        Parameters
        ----------
        x : float or array_like
            x-coordinates to evaluate field at. 
        y : float or array_like
            y-coordinates to evaluate field at.
        beam_pos_x : float
            x-position of beam center.
        beam_pos_y : float
            y-position of beam center.
        pipe_radius : float
            Radius of the beam pipe.
        regular_field : function
            Field for the open-boundary problem, should be a function of (x,y)
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
    r2 = x**2+y**2
    x_im = pipe_radius**2*x/r2
    y_im = pipe_radius**2*y/r2

    direct_field = regular_field(x - beam_pos_x, y - beam_pos_y)
    
    image_field = regular_field(x_im - beam_pos_x, -(y_im - beam_pos_y))                 
    image_field = (-2 + (x_im + 1j*y_im)*image_field)/(x - 1j*y)

    image_field_0 = 2*(beam_pos_x+1j*beam_pos_y)/pipe_radius**2
    image_field = np.where((x==0) & (y==0), image_field_0, image_field)

    inside_field = direct_field + image_field
    outside_field = 2/(x-1j*y)  # Field from point charge
    field = np.where(r2 < pipe_radius**2, inside_field, outside_field)

    if flag_complex_out==False:
        return field.real, field.imag
    else:
        return field

def beam_pipe_gaussian_field(x, y, beam_pos_x, beam_pos_y,
                             sigma_x, sigma_y, pipe_radius,
                             flag_complex_out=False):
    """
        Evaluate E field in beam pipe from Gaussian distribution.

        Evaluate the electric field from a Gaussian beam distribution 
        inside a circular conducting beam pipe using the method of 
        image charges and the Bessetti-Erskine formula. (Wrapper for
        `beam_pipe_field`)

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
        beam_pos_x : float
            x-position of beam center.
        beam_pos_y : float
            y-position of beam center.
        pipe_radius : float
            Radius of the beam pipe.
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
    def regular_field(x,y):
        return gaussian_field(x, y, sigma_x, sigma_y, flag_complex_out=True)
        
    return beam_pipe_field(x, y, beam_pos_x, beam_pos_y, regular_field, 
                           pipe_radius, flag_complex_out)

def beam_pipe_uniform_field(x, y, beam_pos_x, beam_pos_y,
                            a, b, pipe_radius,
                            flag_complex_out=False):
    """
        Evaluate E field in beam pipe from uniform elliptical dist.

        Evaluate the electric field from a uniform elliptical beam 
        distribution inside a circular conducting beam pipe using the
        method of image charges. (Wrapper for `beam_pipe_field`)

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
        beam_pos_x : float
            x-position of beam center.
        beam_pos_y : float
            y-position of beam center.
        pipe_radius : float
            Radius of the beam pipe.
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
    def regular_field(x,y):
        return uniform_field(x, y, a, b, flag_complex_out=True)

    return beam_pipe_field(x, y, beam_pos_x, beam_pos_y, regular_field, 
                           pipe_radius, flag_complex_out)







#def beam_pipe_field(x, y, sigma_x, sigma_y, 
#                    beam_pos_x, beam_pos_y,
#                    pipe_radius, flag_complex_out=False):
#    """
#        Evaluate electric field in beam pipe from Gaussian charge distribution.
#
#        Computes the electric field from a Gaussian beam distribution inside a
#        circular conducting beam pipe using the method of image charges. For points
#        inside the pipe, the field is the sum of the direct beam field and image
#        charge contributions. For points outside the pipe, returns the field from
#        a point charge (infinite pipe approximation).
#
#        Parameters
#        ----------
#        x : float or array_like
#            x-coordinates to evaluate field at. 
#        y : float or array_like
#            y-coordinates to evaluate field at.
#        sigma_x : float
#            Gaussian RMS width in the x direction.
#        sigma_y : float
#            Gaussian RMS width in the y direction.
#        beam_pos_x : float
#            x-position of beam center.
#        beam_pos_y : float
#            y-position of beam center.
#        pipe_radius : float
#            Radius of the beam pipe.
#        flag_complex_out : bool, optional
#            If True, returns complex field (Ex + 1j*Ey).
#            If False, returns separate real and imaginary parts (Ex, Ey).
#            Default is False.
#
#        Returns
#        -------
#        field : complex or tuple
#            If flag_complex_out is True:
#                complex ndarray
#                    Complex field values (Ex + 1j*Ey)
#            If flag_complex_out is False:
#                tuple of (Ex, Ey) where:
#                Ex : ndarray
#                    Real part of field (x-component)
#                Ey : ndarray 
#                    Imaginary part of field (y-component)
#    """
#    r2 = x**2+y**2
#    x_im = pipe_radius**2*x/r2
#    y_im = pipe_radius**2*y/r2
#
#    direct_field = gaussian_field(x - beam_pos_x, y - beam_pos_y,
#                                  sigma_x, sigma_y, flag_complex_out=True)
#    
#    image_field = gaussian_field(x_im - beam_pos_x, -(y_im - beam_pos_y),
#                                 sigma_x, sigma_y, flag_complex_out=True)                 
#    image_field = (-2 + (x_im + 1j*y_im)*image_field)/(x - 1j*y)
#
#    image_field_0 = 2*(beam_pos_x+1j*beam_pos_y)/pipe_radius**2
#    image_field = np.where((x==0) & (y==0), image_field_0, image_field)
#
#    inside_field = direct_field + image_field
#    outside_field = 2/(x-1j*y)  # Field from point charge
#    field = np.where(r2 < pipe_radius**2, inside_field, outside_field)
#
#    if flag_complex_out==False:
#        return field.real, field.imag
#    else:
#        return field