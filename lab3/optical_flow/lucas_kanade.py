import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import scipy.signal as scs



def lucas_kanade(img1, img2, window_size=15):

    assert img1.shape == img2.shape, 'Images are not the same shape but {} and {}'.format(img1.shape, img2.shape)

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

    for y in y_idcs:
        for x in x_idcs:
            # get region
            #print('Region from x={} to {} and y={} to {}'.format(x,x+window_size,y,y+window_size))
            reg1 = img1[x:x+window_size,y:y+window_size]
            reg2 = img2[x:x+window_size,y:y+window_size]

            # compute x and y derivative
            # assume that they are the roughly same for both images
            x_kernel = np.array([[-1, 0, 1], [-2,0,2], [-1,0,1]])
            y_kernel = np.array([[-1, 0, 1], [-2,0,2], [-1,0,1]]).T

            I_x = np.zeros_like(reg2)
            I_y = np.zeros_like(reg2)

            if len(reg2.shape) == 3:
                for i in range(len(reg2.shape)):
                    I_x[:,:,i] = scs.convolve2d(reg2[:,:,i], x_kernel, 'same')
                    I_y[:,:,i] = scs.convolve2d(reg2[:,:,i], y_kernel, 'same')

            else:
                I_x = scs.convolve2d(reg2, x_kernel, 'same')
                I_y = scs.convolve2d(reg2, y_kernel, 'same')
            
            
            # compute time difference:
            I_t = reg2 - reg1

            # flatten into vector
            I_t = I_t.flatten()
            I_x = I_x.flatten()
            I_y = I_y.flatten()

            # compute A and b
            A = np.concatenate((I_x[:,None], I_y[:,None]), axis=1)
            #print('A.shape = ', A.shape)
            b = -1 * I_t

            # compute pseudo-inverse
            A_dagger = np.linalg.inv(A.T @ A) @ A.T

            # compute flow
            flow = A_dagger @ b

            flows.append(flow)
    
    return np.array(flows)


def plot_flows(flows, coarse=True, window_size=15, result_file='./flow_quiver.pdf', quiver_kwargs={}, plot_title=''):

    # get number of windows per column and row
    win_per_col = int(np.sqrt(flows.shape[0]))
    win_per_row = int(np.sqrt(flows.shape[0]))


    # by default an arrow in [1,1] direction will point to the upper right in plt.quiver
    # Our implementation yields arrows that presume to point to the lower right 
    # so we have to rotate counter-clockwise by 90 degrees, which we achieve with a 
    # rotation matrix R
    R = np.array([[0,-1], [1,0]])
    flows = np.einsum('ik,jk',flows,R)

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
    colors = np.arctan2(V, U)
    norm = mpl.colors.Normalize()
    norm.autoscale(colors)
    colormap = mpl.cm.viridis

    # plot results
    plt.figure()
    plt.gca().invert_yaxis()
    color = colormap(norm(colors)).reshape((-1, 4))
    q = plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', color=color, **quiver_kwargs)
    plt.colorbar()
    plt.title(plot_title)
    
    if coarse:
        plt.savefig('./results/coarse_grained/'+result_file)

    else:
        plt.savefig('./results/fine_grained/'+result_file)


def demo():

    # coarse or fine grained plot?
    # fine means that we set an arrow at every pixel
    # coarse means we set only one arrow for every window
    coarse = True

    # compute for sphere
    path_1 = './sphere1.ppm'
    path_2 = './sphere2.ppm'

    # loag images
    img1 = np.array(cv2.imread(path_1, -1), dtype=np.float64) / 255
    img2 = np.array(cv2.imread(path_2, -1), dtype=np.float64) / 255

    # call lucas_kanade
    sphere_flows = lucas_kanade(img1, img2)

    # plot sphere flows
    quiver_kwargs_sphere = {'scale':0.04, 'minlength':0.2, 'headwidth':1.5}
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
    quiver_kwargs_synth = {'scale':0.01, 'minlength':0.2, 'headwidth':3}
    plot_flows(synth_flows, coarse=coarse, window_size=15, result_file='./synth_flow.pdf', quiver_kwargs=quiver_kwargs_synth, plot_title=r'Synth - color $\leftrightarrow$ angle in radiants')


if __name__ == "__main__":
    demo()