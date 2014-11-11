import numpy as np
# import matplotlib.pyplot as plt
#from pylab import pcolor, show, colorbar, xticks, yticks
import scipy

### Function to generate the 1D covariances matrix
#   The covariance is given by the negative squared exponential: Cov(x_i,x_j) = sigma exp(||x_i-x_j||^2 / l)
#   Arguments:
#   Dim: Image Dimension
#   sigma: variance parameter of the covariance matrix
#   l: length scale of the covariance parameter

def generateCov(Dim, sigma, l):
    Cov1D = np.zeros((Dim, Dim), dtype='double')
    for i in range(0, Dim):
        for j in range(0, Dim):
            Cov1D[i][j] = sigma * np.exp(-((i - j) * (i - j) / l))  # + np.random.normal( mean_noise , var_noise)
    return (Cov1D)


epsilon = 1


### This is a very simple function for generating random 2D displacement fields
#   iid Gaussian noise (mean: 0, variance: len) is first generated and then spatial correlation is introduced by Gaussian convolution
#   Function's arguments:
#   Dim: dimension of the image
#   sigma: sigma of the Gaussian convolution
#   len: variance of the iid Gaussian noise


def generate2D_warp(Dim, sigma, len):
    warp = np.append(np.array(range(Dim * Dim), dtype='double'), np.array(range(Dim * Dim), dtype='double')).reshape(2,
                                                                                                                     Dim * Dim)
    # Initialize the warp as random iid noise
    warp[1] = np.random.normal(0, len, Dim * Dim)
    warp[0] = np.random.normal(0, len, Dim * Dim)

    # Define the Gaussian kernel
    G = scipy.signal.gaussian(10, sigma, sym=True)

    # Smooth the warp in both directions (this is not really correct since correlation across boundaries is introduced...)
    warp[0] = scipy.signal.convolve(warp[0], G)[0:Dim * Dim]
    warp[1] = scipy.signal.convolve(warp[1], G)[0:Dim * Dim]

    # Generate a displacement by adding identity at every voxel
    for i in np.arange(0, Dim * Dim):
        warp[0][i] += int(i / Dim)
        warp[1][i] += i % Dim

    return (warp)


### Function to generate a 2D displacement field from the Kronecker product of 1D covariances matrices
#   The covariance is given by the negative squared exponential: Cov(x_i,x_j) = sigma exp(||x_i-x_j||^2 / l)
#   Arguments:
#   Dim: Image Dimension
#   sigma: variance parameter of the covariance matrix
#   len: length scale of the covariance parameter

def generate2D_warpSmooth(Dim, sigma, l):
    #Generate the 1D covariance matrix Sigma
    Cov1D = generateCov(Dim, sigma, l)
    # Compute the Cholesky decomposition of Sigma = C * C.transpose()
    Cholesky = np.linalg.cholesky(Cov1D)
    warp = np.append(np.array(range(Dim * Dim), dtype='double'), np.array(range(Dim * Dim), dtype='double')).reshape(2,
                                                                                                                     Dim * Dim)
    # Generate iid ~ N(0,1) distribuited signal
    Sample0 = np.random.multivariate_normal(np.zeros((1, Dim)), 1, Dim * Dim)
    Sample1 = np.random.multivariate_normal(np.zeros((1, Dim)), 1, Dim * Dim)
    # Compute the warp as w = C * z, z~N(0,1)
    # If C=kron(k1, k2), then C * z = kron(k1, k2) * z = k2 * reshape(z) * k1.transpose()
    warp[0] = np.dot(Cholesky, np.dot(Sample0.reshape(Dim, Dim), Cholesky.transpose())).flatten()
    warp[1] = np.dot(Cholesky, np.dot(Sample1.reshape(Dim, Dim), Cholesky.transpose())).flatten()
    # Generate the displacement field
    for i in np.arange(0, Dim * Dim):
        warp[0][i] += int(i / Dim)
        warp[1][i] += i % Dim
    return (warp)


### This function efficiently computes the data term of the GP based likelihood
# Data Term: (refImg - floatImg).transpose * [Grad(floatImg) * Cov * Grad(floatImg).transpose]^(-1) * (refImg - floatImg)
# where Cov1D = epsilon Id + kronecker(Cov1D, Cov1D)
# The computation is efficient since the product is decomposed in the product of 1 dimensional quantities.
# Parameters:
# refImg,floatImg: reference and floating image respectively
# Cov1D: The 1D covariance matrix
# epsilon: parameter of the iid image intensity noise
#
def DataTerm(refImg, floatImg, Cov1D, epsilon):
    # Compute the gradient of the floating image
    dx = (np.gradient(floatImg)[0].flatten() + 0.001)
    dy = (np.gradient(floatImg)[1].flatten() + 0.001)
    Grad2 = (dx + dy)
    # Get image dimension
    Dim = Cov1D.shape[0]
    # Compute (refImg - floatImg)
    vector = (refImg.flatten() - floatImg.flatten())
    # Efficiently compute the Data Term:
    # I used the Taylor expansion for the inverse (epsilon Id + C)^(-1) ~ 1/epsilon Id - 1/epsilon**2 C + 1/epsilon**3 C^2
    # Note that in this way the computation is always performed consistently with the Kronecker product: we do not need to store the n^2 by n^2 Cov matrix
    #
    # First compute the first term ApproxFinal1 = Grad * Cov * Grad * (refImg - floatImg)
    ApproxFinal1 = Grad2 * np.dot(np.dot(Cov1D, (vector * Grad2).reshape(Dim, Dim)), Cov1D.transpose()).flatten()
    # Then compute the second term ApproxFinal2 = Grad * Cov**2 * Grad * (refImg - floatImg)
    ApproxFinal2 = Grad2 * np.dot(Cov1D, np.dot((Grad2 * Grad2 * np.dot(
        np.dot(Cov1D, (vector * Grad2).reshape(Dim, Dim)), Cov1D.transpose()).flatten()).reshape(Dim, Dim),
                                                Cov1D.transpose())).flatten()
    # Finally compute the data term vector * (1/epsilon - 1/epsilon**2 ApproxFinal1 + 1/epsilon**3 ApproxFinal2)
    return (np.dot(vector, vector / epsilon - 1 / epsilon ** 2 * ApproxFinal1 + 1 / (epsilon ** 3) * ApproxFinal2))

### This function efficiently computes the regularity term of the GP based likelihood
# Regularity Term:  det([Grad(floatImg) * Cov * Grad(floatImg).transpose](-1))
# where Cov1D = epsilon Id + kronecker(Cov1D, Cov1D)
# The computation is efficient since the product is decomposed in the product of 1 dimensional quantities.
# Parameters:
# floatImg: floating image
# Cov1D: The 1D covariance matrix
# epsilon: parameter of the iid image intensity noise
#
def Regularity(Cov1D, epsilon, floatImg):
    # Compute the gradient of the floating image
    dx = (np.gradient(floatImg)[0].flatten() + 0.001)
    dy = (np.gradient(floatImg)[1].flatten() + 0.001)
    Grad2 = (dx + dy)
    trace = 0
    # The matrix inversion is approximated by Taylor expansion: (epsilon Id + C)^(-1) ~ epsilon * trace(C)
    Dim = Cov1D.shape[0]
    for i in range(0, Cov1D.shape[0]):
        trace += sum(Cov1D[i, i] * Cov1D.diagonal() * (Grad2[i * Dim:i * Dim + Dim] * Grad2[i * Dim:i * Dim + Dim]))
    return (epsilon ** (Dim) * (1 + trace))


# This is a snippet for generating random displacements to optimize the likelihood for simple test images
# The attempt does not bring anywhere, but at least it's a sort of test of the code...

Dim = 100
mean_noise = 0
var_noise = 0.03

# Generate simple test images

floatImg = np.zeros((Dim, Dim))
refImg = np.zeros((Dim, Dim))
floatImg[Dim / 3:Dim - Dim / 3, Dim / 3:Dim - Dim / 3] = 1
refImg[Dim / 4:Dim - Dim / 4, Dim / 4:Dim - Dim / 4] = 1

# Generate the 1D covariance matrix

Cov1D = np.zeros((Dim, Dim))
Cov1D = generateCov(Dim, 2, 2)

finalwarp = generate2D_warp(Dim, 4, 2)
warpedImage = scipy.ndimage.interpolation.map_coordinates(floatImg, finalwarp).reshape(Dim, Dim)
OldlogLike = -np.log(Likelihood(refImg, warpedImage, Cov1D, epsilon)) - np.log(Regularity(Cov1D, epsilon, warpedImage))
ntrials = 500

print 'Initial SSD: ', sum((floatImg - refImg).flatten() * (floatImg - refImg).flatten())

for i in range(1, ntrials):
    warp = generate2D_warp(Dim, 4, 2)
    warpedImage = scipy.ndimage.interpolation.map_coordinates(floatImg, warp).reshape(Dim, Dim)
    logLike = -np.log(DataTerm(refImg, warpedImage, Cov1D, epsilon)) - np.log(Regularity(Cov1D, epsilon, warpedImage))
    if (logLike > OldlogLike):
        finalwarp = warp
        OldlogLike = logLike
        print 'Change at ', i, 'Current SSD: ', sum((warpedImage - refImg).flatten() * (warpedImage - refImg).flatten())


