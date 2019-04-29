import odl
import numpy as np
import matplotlib.pyplot as plt
from LDDMM_gd import LDDMM_gd_solver
#from skimage.measure import compare_ssim, compare_psnr


def ssd(image_a, image_b):
    """Compute the sum of squared differences for two gray-valued images.
    The model is:
        
         .. math:: {1/2}*\int_\Omega |I_0(x) - I_1(x)|^2  dx
    Parameters
    ----------
    image_a : `DiscreteLpElement`
        image data.
    image_b: `DiscreteLpElement`
        image data.
  
    Returns
    -------
    ssd : `float`
        Value of sum of squared differences.
    """
    frobenuis=np.linalg.norm(image_a-image_b)

    return 0.5*frobenuis**2


# --- Give input images --- #

I0name = './pictures/v.png' # 64 * 64 ---> 92
I1name = './pictures/j.png' # 64 * 64

# --- Get digital images --- #

#I0 = plt.imread(I0name).astype('float')[:, :]
#I1 = plt.imread(I1name).astype('float')[:, :]
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[:, :]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[:, :]
# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.discr.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[64, 64],
    dtype='float32', interp='linear')

# Create the ground truth as the given image
reference = rec_space.element(I0)

# Create the template as the given image
template = rec_space.element(I1)


# Show intermiddle results
callback = odl.solvers.CallbackShow(
    'iterates', display_step=5) & \
    odl.solvers.CallbackPrintIteration()

#reference.show('reference')
#template.show('template')

# The parameter for kernel function
sigma = 6.0

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Maximum iteration number
niter = 200

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 10E-7


# Compute the sum of squared differences 
ssd_intial = ssd(template, reference)

# Output the sum of squared differences
print('ssd = {!r}'.format(ssd_intial))

# Give the number of time points
time_itvs = 20

# Compute by LDDMM solver
image_N0, E = LDDMM_gd_solver(
        template, reference,  time_itvs, niter, eps, lamb,
        kernel,callback)
    
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])


# Compute the sum of squared differences between the result and reference image
ssd_result = ssd(rec_result,reference)
print('ssd = {!r}'.format(ssd_result))

# Plot the results of interest
plt.figure(1, figsize=(24, 24))
#plt.clf()

plt.subplot(2, 3, 1)
plt.imshow(template, cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.axis('off')
plt.colorbar()
plt.title('Template')

plt.subplot(2, 3, 2)
plt.imshow(rec_result_1, cmap='bone',
           vmin=np.asarray(rec_result_1).min(),
           vmax=np.asarray(rec_result_1).max()) 
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4))

plt.subplot(2, 3, 3)
plt.imshow(rec_result_2, cmap='bone',
           vmin=np.asarray(rec_result_2).min(),
           vmax=np.asarray(rec_result_2).max()) 
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(2, 3, 4)
plt.imshow(rec_result_3, cmap='bone',
           vmin=np.asarray(rec_result_3).min(),
           vmax=np.asarray(rec_result_3).max()) 
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(2, 3, 5)
plt.imshow(rec_result, cmap='bone',
           vmin=np.asarray(rec_result).min(),
           vmax=np.asarray(rec_result).max()) 
plt.axis('off')
plt.colorbar()
plt.title('Reconstructed by {!r} iters'.format(niter))

plt.subplot(2, 3, 6)
plt.imshow(reference, cmap='bone',
           vmin=np.asarray(reference).min(),
           vmax=np.asarray(reference).max())
plt.axis('off')
plt.colorbar()
plt.title('reference')

plt.figure(2, figsize=(8, 1.5))
plt.plot(E)
plt.ylabel('Energy')
plt.gca().axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='--')
