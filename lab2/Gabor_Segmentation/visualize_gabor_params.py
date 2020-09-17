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
