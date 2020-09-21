import numpy as np


def L(s,x,y): 
    '''
    compute the LoG at a specific location in space
    '''

    return(-1/(np.pi*s**4) * np.exp(-1/(2 * s**2 ) * (x**2 + y**2)) * (1 - (x**2 + y**2) / (2*s**2)))


def compute_LoG(n,s):
    '''
    computes 'exact' nxn LoG with std s centered around [n/2,n/2]
    '''
    zero = int((n-1)/2)

    out = np.zeros((n,n))

    for i in range(-zero,zero+1,1):
        for j in range(-zero,zero+1,1):
            out[i+zero,j+zero] = L(s,i,j)
    return out



if __name__ == "__main__":
    a = compute_LoG(5,0.5)
    print(a)

