import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import scipy.signal as scs



def lucas_kanade(img1, img2, window_size=15):
    '''
    performs the lucas kanade algorithm to determine the optical flow from img1 to img2

    returns a column-major ordered list of flows for every region
    '''
    assert img1.shape == img2.shape, 'Images are not the same shape but {} and {}'.format(img1.shape, img2.shape)

    # for numerical stability of matrix inverse
    eps = 1e-10

    # compute how many rows and columns we have to drop
    drop_x = img1.shape[0] % window_size
    drop_y = img1.shape[0] % window_size

    # drop the bottom rows and right columns that are left over
    print('Dropping the {} bottom rows and the {} right columns.'.format(drop_x, drop_y))
    img1 = img1[:-drop_x, :-drop_y]
    img2 = img2[:-drop_x, :-drop_y]

    # getting splitting idcs
    x_idcs = np.arange(0,img1.shape[0],window_size)
    y_idcs = np.arange(0,img1.shape[1],window_size)

    # list to store flows
    flows = []

    # compute derivatives before splitting into regions to have less 
    # errors from boundaries
    # assume that the derivatives are the roughly same for both images
    x_kernel = np.array([[1, 0, -1], [2,0,-2], [1,0,-1]])
    y_kernel = np.array([[1, 0, -1], [2,0,-2], [1,0,-1]]).T

    I_x = np.zeros_like(img1)
    I_y = np.zeros_like(img1)

    if len(img1.shape) == 3:
        for i in range(len(img1.shape)):
            I_x[:,:,i] = scs.convolve2d(img1[:,:,i], x_kernel, 'same', 'symm')
            I_y[:,:,i] = scs.convolve2d(img1[:,:,i], y_kernel, 'same', 'symm')

    else:
        I_x = scs.convolve2d(img1, x_kernel, 'same', 'symm')
        I_y = scs.convolve2d(img1, y_kernel, 'same', 'symm')
    
    I_t = img2 - img1

    for y in y_idcs:
        for x in x_idcs:
            # get region
            #print('Region from x={} to {} and y={} to {}'.format(x,x+window_size,y,y+window_size))
            reg_x = I_x[x:x+window_size,y:y+window_size]
            reg_y = I_y[x:x+window_size,y:y+window_size]
            reg_t = I_t[x:x+window_size,y:y+window_size]
            
            # flatten into vector
            reg_x = reg_x.flatten()
            reg_y = reg_y.flatten()
            reg_t = reg_t.flatten()

            # compute A and b
            A = np.concatenate((reg_x[:,None], reg_y[:,None]), axis=1)
            #print('A.shape = ', A.shape)
            #print('A^TA = ',A.T@A)
            b = -1 * reg_t

            # compute pseudo-inverse
            A_dagger = np.linalg.inv((A.T @ A) + eps*np.eye(2)) @ A.T

            # compute flow
            flow = A_dagger @ b

            flows.append(flow)
    
    return np.array(flows)


def plot_flows(flows, coarse=True, window_size=15, result_file='./flow_quiver.pdf', quiver_kwargs={}, plot_title=''):

    '''
    used for plotting optical flows

    coarse determines whether to plot one arrow per region or one arrow per pixel
    '''

    # get number of windows per column and row
    win_per_col = int(np.sqrt(flows.shape[0]))
    win_per_row = int(np.sqrt(flows.shape[0]))


    if coarse:
        # get meshgrid for arrow locations
        X, Y = np.meshgrid(np.arange(win_per_row), np.arange(win_per_col))
        U, V = np.meshgrid(np.zeros(win_per_row), np.zeros(win_per_col))

        for i in range(flows.shape[0]):
            
            # get corresponding window idcs
            x_id = (i % win_per_col) 
            y_id = (i // win_per_col) 
            #print('x_id = {}; y_id = {}'.format(x_id, y_id))

            # set all arrows in this window the given optical flow
            U[x_id, y_id] = flows[i,0]
            V[x_id, y_id] = flows[i,1]

    else:
        # get meshgrid for arrow locations
        X, Y = np.meshgrid(np.arange(win_per_row* window_size), np.arange(win_per_col* window_size))
        U, V = np.meshgrid(np.zeros(win_per_row* window_size), np.zeros(win_per_col* window_size))

        for i in range(flows.shape[0]):
            
            # get corresponding window idcs
            x_id = (i % win_per_col) * window_size
            y_id = (i // win_per_col) * window_size

            # set all arrows in this window the given optical flow
            U[x_id:x_id+window_size, y_id:y_id+window_size] += flows[i,0]
            V[x_id:x_id+window_size, y_id:y_id+window_size] += flows[i,1]


    # compute colors
    # colors correspond to angle of arrow in radiants
    angles = np.arctan2(V,U)
    
    
    #plt.figure()
    #plt.hist(angles.flatten())
    #plt.show()

    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
    #norm.autoscale(colors)
    colormap = mpl.cm.inferno

    # plot results
    plt.figure()
    plt.gca().invert_yaxis() # put origin in top left corner. Does not affect the direction of arrows unless angles='xy'
    colors = colormap(norm(angles)).reshape((-1, 4))
    q = plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', color=colors, **quiver_kwargs)
    plt.clim(-np.pi,np.pi)
    plt.set_cmap('inferno')
    plt.colorbar()
    plt.title(plot_title)
    
    if coarse:
        plt.savefig('./results/coarse_grained/'+result_file)

    else:
        plt.savefig('./results/fine_grained/'+result_file)
    plt.close()


def demo(coarse=True):
    '''
    computes optical flow for the example images and plots the results
    '''
    # coarse determines whether we display a coarse or fine-grained plot
    # fine means that we set an arrow at every pixel
    # coarse means we set only one arrow for every window

    # compute for sphere
    path_1 = './sphere1.ppm'
    path_2 = './sphere2.ppm'

    # loag images
    img1 = np.array(cv2.imread(path_1, -1), dtype=np.float64) / 255
    img2 = np.array(cv2.imread(path_2, -1), dtype=np.float64) / 255

    # call lucas_kanade
    sphere_flows = lucas_kanade(img1, img2)

    # plot sphere flows
    quiver_kwargs_sphere = {'scale':0.2, 'minlength':0.2, 'headwidth':3}
    plot_flows(sphere_flows, coarse=coarse, window_size=15, result_file='./sphere_flow.pdf', quiver_kwargs=quiver_kwargs_sphere, plot_title=r'Sphere - color $\leftrightarrow$ angle in radiants')

    #########
    #########

    # compute for synth
    path_1 = './synth1.pgm'
    path_2 = './synth2.pgm'

    # loag images
    img1 = np.array(cv2.imread(path_1, -1), dtype=np.float64) / 255
    img2 = np.array(cv2.imread(path_2, -1), dtype=np.float64) / 255

    # call lucas_kanade
    synth_flows = lucas_kanade(img1, img2)

    # plot synth flows
    quiver_kwargs_synth = {'scale':0.1, 'minlength':0.2, 'headwidth':3}
    plot_flows(synth_flows, coarse=coarse, window_size=15, result_file='./synth_flow.pdf', quiver_kwargs=quiver_kwargs_synth, plot_title=r'Synth - color $\leftrightarrow$ angle in radiants')


if __name__ == "__main__":

    # compute for a coarse setting, i.e. one arrow per region
    demo(coarse=True)

    #compute for a fine setting, i.e. one arrow per pixel
    demo(coarse=False)
