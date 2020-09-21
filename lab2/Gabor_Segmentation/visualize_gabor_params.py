import numpy as np
import matplotlib.pyplot as plt
from createGabor import createGabor


# each combination is a tuple (theta, sigma, gamma)
params = [
    (0, 1, 1),
    (np.pi/6, 1, 1),
    (0, 0.5, 1),
    (0, 1, 3)
    ]

plt.figure(figsize=(9, 9))
for i, (theta, sigma, gamma) in enumerate(params):
    gabor = createGabor(sigma, theta, 1, 0, gamma)
    plt.subplot(2, 2, i + 1)
    plt.imshow(gabor[:, :, 1], cmap="gray")
    plt.title("θ = {:.0f}°, σ = {}, γ = {}".format(theta * 360 / (2 * np.pi), sigma, gamma))
plt.savefig("fig/parameter_comparison.pdf")



## alternative experimental plotting
# each combination is a tuple (theta, sigma, gamma)
params = [
    (0, 0.5, 1),
    (0, 1, 1),
    (0, 2, 1),
    (np.pi/6,1,1),
    (np.pi/3,1,1),
    (np.pi/2,1,1),
    (0, 1, 0.5),
    (0, 1, 2),
    (0, 1, 3)
    ]

plt.figure(figsize=(9, 9))
for i, (theta, sigma, gamma) in enumerate(params):
    gabor = createGabor(sigma, theta, 1, 0, gamma)[:,:,0] # only take real part
    
    # pad to 13 x 13
    if gabor.shape[0] != 13:
        out = np.ones((13,13)) * gabor[0,0]
    padding_idx = int((13 - gabor.shape[0]) / 2)
    out[padding_idx:padding_idx+gabor.shape[0], padding_idx:padding_idx+gabor.shape[1]] = gabor
    gabor = out
    
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(3, 3, i + 1)
    plt.imshow(gabor, cmap="gray")
    plt.title("θ = {:.0f}°, σ = {}, γ = {}".format(theta * 360 / (2 * np.pi), sigma, gamma))
plt.savefig("fig/parameter_comparison_new.pdf")


## alternative experimental plotting
# each combination is a tuple (theta, sigma, gamma)
params = [
    (np.pi/2 * 0/8,10,1),
    (np.pi/2 * 1/8,10,1),
    (np.pi/2 * 2/8,10,1),
    (np.pi/2 * 3/8,10,1),
    (np.pi/2 * 4/8,10,1),
    (np.pi/2 * 5/8,10,1),
    (np.pi/2 * 6/8,10,1),
    (np.pi/2 * 7/8,10,1),
    (np.pi/2 * 8/8,10,1)
    ]

plt.figure(figsize=(9, 9))
for i, (theta, sigma, gamma) in enumerate(params):
    gabor = createGabor(sigma, theta, 0.9, 0, gamma)[:,:,0] # only take real part
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(3, 3, i + 1)
    plt.imshow(gabor, cmap="gray")
    plt.title("θ = {:.0f}°, σ = {}, γ = {}".format(theta * 360 / (2 * np.pi), sigma, gamma))
plt.savefig("fig/theta_investigation.pdf")