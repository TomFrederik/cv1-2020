import numpy as np

def construct_surface(p, q, path_type='column'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """
        
        ## this is the pythonic version
        # compute cumulative sum over first column
        q_cumsum = np.cumsum(q[:,0]) 

        # compute height map (checked against double-loop, produces same result)
        height_map = np.cumsum(np.concatenate((q_cumsum[:,None], p[:,1:]), axis=1), axis=1)

        ## this is the brute force variant (we checked that the results are the same)
        '''
        for i in range(h):
            height_map[i,0] = q[i,0]
            if i>0:
                height_map[i,0] += height_map[i-1,0]

        for i in range(h):
            for j in range(1,w):
                height_map[i,j] += p[i,j] + height_map[i,j-1]
        '''

    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
        # compute cumulative sum over first row
        p_cumsum = np.cumsum(p[0,:]) 

        # compute height map (checked against double-loop, produces same result)
        height_map = np.cumsum(np.concatenate((p_cumsum[None,:], q[1:,:]), axis=0), axis=0)

    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """

        ## column path
        # compute cumulative sum over first column
        q_cumsum = np.cumsum(q[:,0]) 

        # compute height map
        col_height_map = np.cumsum(np.concatenate((q_cumsum[:,None], p[:,1:]), axis=1), axis=1)
        
        ## row path
        # compute cumulative sum over first row
        p_cumsum = np.cumsum(p[0,:]) 

        # compute height map (checked against double-loop, produces same result)
        row_height_map = np.cumsum(np.concatenate((p_cumsum[None,:], q[1:,:]), axis=0), axis=0)

        ## average over both paths
        height_map = 0.5 * (col_height_map + row_height_map)

    return height_map
        


