def main():
    import numpy as np
    from PIL import Image
    from matplotlib import pyplot as plt
    #int32 to prevent overflow of the standard uint8 during element-wise multiplication
    original = np.array(Image.open('ball.png'))
    shading = np.array(Image.open('ball_shading.png'), dtype="int32")
    albedo = np.array(Image.open('ball_albedo.png'), dtype="int32")
    print(np.unique(albedo))
    albedogreen = np.where(albedo==108, 0, albedo)
    albedogreen = np.where(albedo==141, 255, albedogreen)
    albedogreen = np.where(albedo==184, 0, albedogreen)
    #for broadcasting shapes
    shading = np.expand_dims(shading,-1)

    #I(x)=R(X) x S(x) for all x in one line
    #/255 to renormalise
    reconstruction = np.array(np.multiply(shading,albedogreen)/256,dtype="uint8")



    fig=plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(original)
    plt.subplot(2, 2, 2)
    plt.imshow(reconstruction)

    plt.show()

if __name__ == "__main__":
    main()
