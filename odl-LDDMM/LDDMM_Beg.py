"""
Shape-based reconstruction by large deformation diffeomorphic metric mapping.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
from odl.discr import (Gradient, ResizingOperator)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
from odl.operator import DiagonalOperator
from odl.deform.linearized import _linear_deform
standard_library.install_aliases()


__all__ = ('LDDMM_Beg_solver',)


def _padded_ft_op(space, padded_size):
    """Create zero-padding Fourier transform.

    Parameters
    ----------
    space : the space needed to do Fourier transform.
    padded_size : the padded size in each axis.

    Returns
    -------
    padded_ft_op : `operator`
        Composed operator of Fourier transform composing with padded operator.
    """
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padded_op.range, halfcomplex=False, shift=shifts, impl='pyfftw')

    return ft_op * padded_op


def _vectorized_kernel(space, kernel):
    """Compute the vectorized discrete kernel ``K``.
    
    Parameters
    ----------
    space : the space used to define kernel ``K``.
    kernel : the used kernel function for data fitting term.
    
    Returns
    -------
    discretized_kernel : `ProductSpaceElement`
        The vectorized discrete kernel with the space dimension
    """
    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel


def LDDMM_Beg_solver( template,reference,  time_pts,
                                  niter, eps, lamb, kernel, callback=None):
    """
    Solver for the shape-based reconstruction using LDDMM_Beg.

    Notes
    -----
    The model is:
                
        .. math:: \min_{v(t) \in V, \phi_{0,1}^v \in G_v}  \lambda \int_0^1 \|v(t)\|_V^2 dt + \int_\Omega |(\phi_{0,1}^v.I_0) - I_1|^2  dx

    Here :math:`I_0` is the template. :math:`v(t)` is the
    velocity vector. :math:`V` is a reproducing kernel Hilbert space for
    the velocity vector.  :math:`I_1` is the reference. :math:`\lambda` is
    the regularization parameter. :math:`\phi_{0, 1}^v.I 
    := I \circ \phi_{1, 0}^v` is for geometric deformation,
    :math:`\phi_{1, 0}^v` is the inverse of
    :math:`\phi_{0, 1}^v` i.e. the solution at :math:`t=1` of flow of
    diffeomorphisms. :math:`|D\phi_{1, 0}^v|` is the Jacobian determinant
    of :math:`\phi_{1, 0}^v`. 
    Parameters
    ----------
    
    template : `DiscreteLpElement`
        Fixed template.
    reference: `DiscreteLpElement`
        Fixed reference.
    time_pts : `int`
        The number of time intervals
    niter : `int`
        The given maximum iteration number.
    eps : `float`
        The given step size.
    lamb : `float`
        The given regularization parameter. It's a weighted value on the
        regularization-term side.
    kernel : `function`
        Kernel function in reproducing kernel Hilbert space.
    callback : `class`, optional
        Show the intermediate results of iteration.

    Returns
    -------
    image_N0 : `ProductSpaceElement`
        The series of images produced by template and velocity field.
    E : `numpy.array`
        Storage of the energy values for iterations.
    """

    # Give the number of time intervals
    N = time_pts

    # Get the inverse of time intervals
    inv_N = 1.0 / N

    # Create the space of images
    image_space = template.space
    #image_space= rec_space

    # Get the dimension of the space of images
    dim = image_space.ndim
    
    # Fourier transform setting for data matching term
    # The padded_size is the size of the padded domain 
    padded_size = 2 * image_space.shape[0]
    # Create operator of Fourier transform composing with padded operator
    pad_ft_op = _padded_ft_op(image_space, padded_size)
    # Create vectorial Fourier transform operator
    # Construct the diagnal element of a matrix operator
    vectorial_ft_op = DiagonalOperator(*([pad_ft_op] * dim))
    
    # Compute the FT of kernel in fitting term
    discretized_kernel = _vectorized_kernel(image_space, kernel)
    ft_kernel_fitting = vectorial_ft_op(discretized_kernel)

    # Create the space for series deformations and series Jacobian determinant
    pspace = image_space.tangent_bundle
    series_pspace = ProductSpace(pspace, N+1)
    series_image_space = ProductSpace(image_space, N+1)

    # Initialize vector fileds at different time points
    vector_fields = series_pspace.zero()

    # Give the initial two series deformations and series Jacobian determinant
    image_N0 = series_image_space.element()
    image_N1 = series_image_space.element()
   
    
    detDphi_N1 = series_image_space.element()
    
    for i in range(N+1):
        image_N0[i] = image_space.element(template)
        image_N1[i] = image_space.element(reference)
     
        detDphi_N1[i] = image_space.one()

    # Create the gradient operator
    grad_op = Gradient(domain=image_space, method='forward',
                       pad_mode='symmetric')

    # Create the divergence operator, which can be obtained from
    # the adjoint of gradient operator 
    # div_op = Divergence(domain=pspace, method='forward', pad_mode='symmetric')
    div_op = -grad_op.adjoint
    
    # Store energy
    E = []
    kE = len(E)
    E = np.hstack((E, np.zeros(niter)))

    # Begin iteration 

    for k in range(niter):
            # Update the velocity field
            for i in range(N+1):
                #first term
                term1_tmp1=2*np.abs(image_N0[i]-image_N1[i])*(detDphi_N1[i])
                term1_tmp = grad_op(image_N0[i])
              
            for j in range(dim):            
                term1_tmp[j] *=term1_tmp1
               
            tmp3 = (2 * np.pi) ** (dim / 2.0) * vectorial_ft_op.inverse(
                    vectorial_ft_op(term1_tmp) * ft_kernel_fitting)
    
            vector_fields[i] = (vector_fields[i] - eps * (
                    lamb * vector_fields[i] - tmp3))
    
            # Update image_N0,image_N1 and detDphi_N0 detDphi_N1
            for i in range(N):
                # Update image_N0[i+1] by image_N0[i] and vector_fields[i+1]
                image_N0[i+1] = image_space.element(
                    _linear_deform(image_N0[i],
                                   -inv_N * vector_fields[i+1]))

                
               # Update image_N1[N-i-1] by image_N1[N-i] and vector_fields[N-i-1]
                image_N1[N-i-1] = image_space.element(
                    _linear_deform(image_N1[N-i],
                                   inv_N * vector_fields[N-i-1]))      
                
#                # Update detDphi_N1[N-i-1] by detDphi_N1[N-i]
#                jacobian_det = image_domain.element(
#                    np.exp(inv_N * div_o  p(vector_fields[N-i-1])))
                jacobian_det = image_space.element(
                        1.0 + inv_N * div_op(vector_fields[N-i-1]))
                detDphi_N1[N-i-1] = (
                    jacobian_det * image_space.element(_linear_deform(
                        detDphi_N1[N-i], inv_N * vector_fields[N-i-1])))
            
            # Update the deformed template
            PhiStarI = image_N0[N]
    
            # Show intermediate result
            if callback is not None:
                callback(PhiStarI)

            # Compute the energy of the data fitting term 
            E[k+kE] += np.asarray((PhiStarI - reference)**2).sum()

    return image_N0, E

    
if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
    