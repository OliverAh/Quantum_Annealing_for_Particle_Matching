is_cupy = False
try:
    import cupy as np
    #import numpy as np
    print('imported cupy')
    is_cupy = True
except:
    print('no cupy available, imported numpy as usual')
    import numpy as np


def calc_phi_ij(coords_n, coords_n_minus_1, type = 'euclidean', sqroot = True, eps=1.0, tanh_percentile=20, tanh_min=0.0): #tanh_percentile=(0,100), tanh_min=(0,inf)
    #print('Compute distance function of {} particles'.format(num_particles))
    phi_ij = np.zeros((len(coords_n), len(coords_n_minus_1)))
    if type == 'euclidean':
        #print('type = euclidean')
        for i in range(np.shape(coords_n)[1]): # loop over x,y,z
            phi_ij += np.square(np.subtract.outer(coords_n[:,i], coords_n_minus_1[:,i]))
        if sqroot == True:
            #print('compute root')
            phi_ij = np.sqrt(phi_ij)
    
    elif type == 'euclidean_eps':
        #print('type = euclidean_eps')
        for i in range(np.shape(coords_n)[1]): # loop over x,y,z
            phi_ij += np.square(np.subtract.outer(coords_n[:,i], coords_n_minus_1[:,i]))
        if sqroot == True:
            #print('compute root')
            phi_ij = np.sqrt(phi_ij)
        phi_ij = -np.exp(-eps*phi_ij) # - in beginning because exp(-eps...) --> large dist -> small phi_ij
        assert False, "distance matrix MUST contain ONLY values > 0 because Q_1 basically is dist^2, large negative would still become large positive"
    
    elif type == 'tanh_percentile':
        #print('type = tanh')
        for i in range(np.shape(coords_n)[1]): # loop over x,y,z
            phi_ij += np.square(np.subtract.outer(coords_n[:,i], coords_n_minus_1[:,i]))
        if sqroot == True:
            #print('compute root')
            phi_ij = np.sqrt(phi_ij)
        print('max: ', np.max(phi_ij))
        print('tanh_percentile: ', tanh_percentile)
        print('percentile: ', np.percentile(phi_ij, tanh_percentile))
        phi_ij = phi_ij / np.percentile(phi_ij, tanh_percentile)
        phi_ij = np.tanh(phi_ij)
    
    elif type == 'tanh_minimum':
        #print('type = tanh')
        for i in range(np.shape(coords_n)[1]): # loop over x,y,z
            phi_ij += np.square(np.subtract.outer(coords_n[:,i], coords_n_minus_1[:,i]))
        if sqroot == True:
            #print('compute root')
            phi_ij = np.sqrt(phi_ij)
        print('min: ', np.min(phi_ij))
        print('tanh_min: ', tanh_min)
        phi_ij = phi_ij / np.min(phi_ij) * tanh_min
        print('min: ', np.min(phi_ij))
        phi_ij = np.tanh(phi_ij)
        print('min: ', np.min(phi_ij))
    
    
    else:
        print('Type {} for distance matrix is not available.'.format(type))
    return(phi_ij)