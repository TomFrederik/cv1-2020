def main():
    import numpy as np
    from PIL import Image
    from matplotlib import pyplot as plt
    #int32 to prevent overflow of the standard uint8 during element-wise multiplication
    original = np.array(Image.open('ball.png'))
    print(np.max(original))
    shading = np.array(Image.open('ball_shading.png'), dtype="int32")
    albedo = np.array(Image.open('ball_albedo.png'), dtype="int32")
    #for broadcasting shapes
    shading = np.expand_dims(shading,-1)

    #I(x)=R(X) x S(x) for all x in one line
    #/255 to renormalise
    reconstruction = np.array(np.multiply(shading,albedo)/256,dtype="uint8")
    print(np.linalg.norm(original-reconstruction))

    fig=plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(shading,cmap='gray',vmin=0,vmax=255)
    plt.title('Shading')
    plt.subplot(2, 2, 3)
    plt.imshow(albedo)
    plt.title('Albedo')
    plt.subplot(2, 2, 4)
    plt.imshow(reconstruction)
    plt.title('Reconstruction')

    plt.show()

if __name__ == "__main__":
    main()
