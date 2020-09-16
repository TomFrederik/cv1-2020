import numpy as np
import matplotlib.pyplot as plt

def createGabor( sigma, theta, lamda, psi, gamma ):
#CREATEGABOR Creates a complex valued Gabor filter.
#   myGabor = createGabor( sigma, theta, lamda, psi, gamma ) generates
#   Gabor kernels.  
#   - ARGUMENTS
#     sigma      Standard deviation of Gaussian envelope.
#     theta      Orientation of the Gaussian envelope. Takes arguments in
#                the range [0, pi/2).
#     lamda     The wavelength for the carriers. The central frequency 
#                (w_c) of the carrier signals.
#     psi        Phase offset for the carrier signal, sin(w_c . t + psi).
#     gamma      Controls the aspect ratio of the Gaussian envelope
#   
#   - OUTPUT
#     myGabor    A matrix of size [h,w,2], holding the real and imaginary 
#                parts of the Gabor in myGabor(:,:,1) and myGabor(:,:,2),
#                respectively.
                
    # Set the aspect ratio.
    sigma_x = sigma
    sigma_y = sigma/gamma

    # Generate a grid
    nstds = 3
    xmax = max(abs(nstds*sigma_x*np.cos(theta)),abs(nstds*sigma_y*np.sin(theta)))
    xmax = np.ceil(max(1,xmax))
    ymax = max(abs(nstds*sigma_x*np.sin(theta)),abs(nstds*sigma_y*np.cos(theta)))
    ymax = np.ceil(max(1,ymax))

    # Make sure that we get square filters. 
    xmax = max(xmax,ymax)
    ymax = max(xmax,ymax)
    xmin = -xmax 
    ymin = -ymax

    # Generate a coordinate system in the range [xmin,xmax] and [ymin, ymax]. 
    [x,y] = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

    # Convert to a 2-by-N matrix where N is the number of pixels in the kernel.
    XY = np.concatenate((x.reshape(1, -1), y.reshape(1, -1)), axis=0)

    # Compute the rotation of pixels by theta.
    # \\ Hint: Use the rotation matrix to compute the rotated pixel coordinates: rot(theta) * XY.
    rotMat = generateRotationMatrix(theta)
    rot_XY = np.matmul(rotMat,XY)
    rot_x = rot_XY[0,:]
    rot_y = rot_XY[1,:]


    # Create the Gaussian envelope.
    # \\ IMPLEMENT the helper function createGauss.
    gaussianEnv = createGauss(rot_x, rot_y, gamma, sigma)

    # Create the orthogonal carrier signals.
    # \\ IMPLEMENT the helper functions createCos and createSin.
    cosCarrier = createCos(rot_x, lamda, psi)
    sinCarrier = createSin(rot_x, lamda, psi)

    # Modulate (multiply) Gaussian envelope with the carriers to compute 
    # the real and imaginary components of the omplex Gabor filter. 
    myGabor_real = gaussianEnv * cosCarrier
    myGabor_imaginary = gaussianEnv * sinCarrier

    # Pack myGabor_real and myGabor_imaginary into myGabor.
    h, w = myGabor_real.shape
    myGabor = np.zeros((h, w, 2))
    myGabor[:,:,0] = myGabor_real
    myGabor[:,:,1] = myGabor_imaginary

    # Uncomment below line to see how are the gabor filters
    # figure
    # subplot(121), imshow(myGabor_real,[])
    # subplot(122), imshow(myGabor_imaginary, [])

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(myGabor_real)    # Real
    ax.axis("off")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(myGabor_imaginary)    # Real
    ax.axis("off")
    plt.show()

    return myGabor


# Helper Functions 
# ----------------------------------------------------------
def generateRotationMatrix(theta):
    # ----------------------------------------------------------
    # Returns the rotation matrix. 
    # \\ Hint: https://en.wikipedia.org/wiki/Rotation_matrix \\
    rotMat = [[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]]
    return rotMat

# ----------------------------------------------------------
def createCos(rot_x, lamda, psi):
    # ----------------------------------------------------------
    # Returns the 2D cosine carrier. 
    cosCarrier = np.cos((2 * np.pi * rot_x / lamda + psi))

    # Reshape the vector representation to matrix.
    cosCarrier = np.reshape(cosCarrier, (np.int(np.sqrt(len(cosCarrier))), -1))
    return cosCarrier

# ----------------------------------------------------------
def createSin(rot_x, lamda, psi):
    # ----------------------------------------------------------
    # Returns the 2D sine carrier. 
    sinCarrier = np.sin(2 * np.pi * rot_x / lamda + psi)

    # Reshape the vector representation to matrix.
    sinCarrier = np.reshape(sinCarrier, (np.int(np.sqrt(len(sinCarrier))), -1))
    return sinCarrier

# ----------------------------------------------------------
def createGauss(rot_x, rot_y, gamma, sigma):
    # ----------------------------------------------------------
    # Returns the 2D Gaussian Envelope. 
    gaussEnv = np.exp(-(rot_x ** 2 + (rot_y * gamma) ** 2)/(2 * sigma ** 2))

    # Reshape the vector representation to matrix.
    gaussEnv = np.reshape(gaussEnv, (np.int(np.sqrt(len(gaussEnv))), -1))
    return gaussEnv

