'''
This file contains the functions for the mixing models used to evaluate the mixing quality.
All models are taken from 

@article{case_study_WEN20151630,
title = {Comparative Study on the Characterization Method of Particle Mixing Index Using DEM Method},
journal = {Procedia Engineering},
volume = {102},
pages = {1630-1642},
year = {2015},
note = {New Paradigm of Particle Science and Technology Proceedings of The 7th World Congress on Particle Technology},
issn = {1877-7058},
doi = {https://doi.org/10.1016/j.proeng.2015.01.299},
url = {https://www.sciencedirect.com/science/article/pii/S1877705815003185},
author = {Yuanyun Wen and Malin Liu and Bing Liu and Youlin Shao},
keywords = {Particle mixing index, DEM, Characterization method, Comparative study}
}
'''


import numpy as np
from src.particle_funcs.distance_matrix import calc_phi_ij


def average_height(coords, axis, species, normalize=0, verbose=0):
    '''
    This function calculates the average height of the particles in the specified axis.
    The species array is used to specify which particles are in the same species.
    '''
    # Get the unique species, and their number
    species_unique = np.unique(species)
    num_species = len(species_unique)
    if verbose != 0:
        print('Number of, and uniqe species: ', num_species, species_unique)
    # Initialize the array to store the average height
    avg_height = np.zeros(num_species)
    # Loop over the species
    for i in range(num_species):
        coords_to_avg = coords[np.where(species == species_unique[i]),axis] # Could be more than one species, then we can not split it binary.
        avg_height[i] = np.mean(coords_to_avg)
    if normalize != 0:
        avg_height /= np.mean(coords[:,axis])
    return avg_height


def nearest_neighbour(coords, species, num_neighbours=12, kwargs_distance_matrix={}, verbose=0):
    '''
    This function calculates the average number of nearest neighbours, of a different species. The default value of 12 nearest neighbours to consider comes from the literature, and can be changed.
    '''
    # Get the unique species, and their number
    species_unique = np.unique(species)
    num_species = len(species_unique)
    if verbose != 0:
        print('Number of, and uniqe species: ', num_species, species_unique)
    # Initialize the array to store the average number of nearest neighbours
    avg_neighbours = np.zeros(num_species)
    # Loop over the species
    dist_matrix = calc_phi_ij(coords, coords, **kwargs_distance_matrix)
    for i in range(num_species):
        # Get the distance matrix for the species in the form of a 2D array and of size (num_particles of this species, num_particles in total)
        dist_matrix_to_sort = dist_matrix[np.where(species==species_unique[i])[0],:]
        # Get the indices of the k nearest neighbours of each particle, this is the k nearest for each row.
        indices_of_k_nearest = np.argpartition(dist_matrix_to_sort, kth=num_neighbours+1, axis=1)[:,1:num_neighbours+1] # +1 because the first index is the particle itsef, and the distance of this is 0
        #print('shapes, dist_mat_to_sort, indices_sorted ', dist_matrix.shape, dist_matrix_to_sort.shape, indices_of_k_nearest.shape)
        #print('indices_of_k_nearest: ', indices_of_k_nearest)
        
        # The average number of nearest neighbours is computed by the number of nearest neighbours of ALL DIFFERENT species, divided by the number of particles of the current species.
        avg_neighbours[i] = len(np.where(species[indices_of_k_nearest]!=species_unique[i])[0]) # is the sum over all particles already, because np.where returns a tuple of arrays, where each holds one index of the of the full multidimensional indexing
        avg_neighbours[i] /= len(np.where(species==species_unique[i])[0])
                ####### does the same, but with a loop
        #for j in range(indices_of_k_nearest.shape[0]):
        #    indices_of_nearest_not_species = np.where(species[indices_of_k_nearest[j,:]]!=species_unique[i])
        #    print(indices_of_nearest_not_species[0])
        #    avg_neighbours[i] += len(indices_of_nearest_not_species[0])
        #avg_neighbours[i] /= indices_of_k_nearest.shape[0]
    return avg_neighbours


# function to compute Lacey index
def calc_lacey_index(coords, species, sample_method='random', sample_count=100, kwargs_distance_matrix={}, verbose=0):
    '''ToDo: implement/apply clustering function'''
    
    # Get the unique species, and their number
    species_unique = np.unique(species)
    num_species = len(species_unique)
    if verbose != 0:
        print('Number of, and uniqe species: ', num_species, species_unique)
    
    if sample_method == 'random':
        sample_origins_index = np.random.default_rng().choice(range(coords.shape[0]), sample_count)
    else:
        print('Could not determine sampling method for Lacey index. Currently supported sampling methods: \'random\'.')
    
    # Get distance matrix to assign all particles to their closest cluster/cell
    distance_matrix = calc_phi_ij(coords, coords[sample_origins_index], **kwargs_distance_matrix)
    #print(distance_matrix.shape)

    # Get indices within distance matrix of the closest clusters for each particle
    clustering = np.argpartition(distance_matrix, kth=2, axis=1)
    #print(clustering.shape)
    #print(clustering)
    clustering = np.where(sample_origins_index[clustering[:,0]] != range(coords.shape[0]), clustering[:,0], clustering[:,1])
    #print(sample_origins_index)
    #print(sample_origins_index[clustering])

    average_number_of_parts_in_cell = coords.shape[0]/sample_count
    average_fraction_of_parts_in_species = np.zeros(num_species)
    for i in range(num_species):
        average_fraction_of_parts_in_species[i] = len(np.where(species==species_unique[i])[0])/coords.shape[0]
    #print(average_fraction_of_parts_in_species)
    var_0 = np.multiply(average_fraction_of_parts_in_species, 1-average_fraction_of_parts_in_species)
    var_r = var_0/average_number_of_parts_in_cell
    #print(var_0, var_r)
    
    lacey_indices = np.zeros(num_species)
    for j in range(num_species):
        var = 0.0
        for i in range(sample_count):
            num_parts_in_cell = len(np.where(clustering==i)[0]) + 1 # +1 because initial particle itself is also in cell
            num_parts_species_in_cell = len(np.where(species[np.where(clustering==i)[0]]==species_unique[j])[0])
            fraction_parts_species_in_cell = num_parts_species_in_cell/num_parts_in_cell
            var += (fraction_parts_species_in_cell - average_fraction_of_parts_in_species[j])**2
        var /= sample_count - 1
        #print(var)
        lacey_indices[j] = (var_0[j] - var)/(var_0[j] - var_r[j])

    return lacey_indices


def calc_mixing_entropy(coords, species, sample_method='random', sample_count=100, method='combined', kwargs_distance_matrix={}, verbose=0, **kwargs):
    '''Is highly sensitive to clusters (placement, and numbers).'''
    if method == 'combined':
        pass
    else:
        print('Could not determine mixing entropy method. Currently supported methods: \'combined\'.')
        return None
    if sample_method == 'random':
        sample_origins_index = np.random.default_rng().choice(range(coords.shape[0]), sample_count)
    
        # Get distance matrix to assign all particles to their closest cluster/cell
        distance_matrix = calc_phi_ij(coords, coords[sample_origins_index], **kwargs_distance_matrix)
        #print(distance_matrix.shape)

        # Get indices within distance matrix of the closest clusters for each particle
        clustering = np.argpartition(distance_matrix, kth=2, axis=1)
        clustering = np.where(sample_origins_index[clustering[:,0]] != range(coords.shape[0]), clustering[:,0], clustering[:,1])
        clusters = []
        for i in range(sample_count):
            clusters.append(np.where(clustering==i)[0])
    
    elif sample_method == 'equidistant_bounds':
        ''' Required kwargs: bounds = {x: [x_min, x_max], y: [y_min, y_max], z: [z_min, z_max]}, sample_counts = {x: x_count, y: y_count, z: z_count]'''
        
        _x = np.linspace(kwargs['bounds']['x'][0], kwargs['bounds']['x'][1], kwargs['sample_counts']['x'])
        _y = np.linspace(kwargs['bounds']['y'][0], kwargs['bounds']['y'][1], kwargs['sample_counts']['y'])
        _z = np.linspace(kwargs['bounds']['z'][0], kwargs['bounds']['z'][1], kwargs['sample_counts']['z'])
        grid_x, grid_y, grid_z = np.meshgrid(_x, _y, _z, indexing='ij')

        sample_origins = np.array([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T
        
        # Get distance matrix to assign all particles to their closest cluster/cell
        distance_matrix = calc_phi_ij(coords, sample_origins, **kwargs_distance_matrix)
        #print(distance_matrix.shape)

        # Get indices within distance matrix of the closest clusters for each particle
        clustering = np.argpartition(distance_matrix, kth=1, axis=1)
        #clustering = np.where(sample_origins_index[clustering[:,0]] != range(coords.shape[0]), clustering[:,0], clustering[:,1]) # not needed here
        clustering = clustering[:,0]
        clusters = []
        for i in range(sample_count):
            clusters.append(np.where(clustering==i)[0])


    else:
        print('Could not determine sampling method for mixing entropy. Currently supported sampling methods: \'random\'.')
        return None
    
    
    cluster_sizes = np.array([len(clusters[i]) for i in range(sample_count)])
    cluster_fractions = cluster_sizes/coords.shape[0]
    cluster_fractions_species = np.zeros((sample_count, len(species)))
    #print(clusters)
    #print(cluster_sizes)

    for i in range(sample_count):
        unique, counts = np.unique(species[clusters[i]], return_counts=True)
        cluster_fractions_species[i, unique.astype(int)] = counts
        cluster_fractions_species /= cluster_sizes[i] if cluster_sizes[i] != 0 else 1

    local_entropies = np.zeros(sample_count)
    for i in range(len(clusters)):
        #print(cluster_fractions_species[i])
        #print(np.log(cluster_fractions_species[i]))
        #print(np.multiply(cluster_fractions_species[i], np.log(cluster_fractions_species[i])))
        _tmp = np.where(cluster_fractions_species[i] == 0, 0, np.multiply(cluster_fractions_species[i], np.log(cluster_fractions_species[i])))
        #print(_tmp)
        local_entropies[i] = np.sum(_tmp)

        #local_entropies[i] = np.sum(np.multiply(cluster_fractions_species[i], np.log(cluster_fractions_species[i])))

    global_entropy = np.sum(np.multiply(local_entropies, cluster_sizes))
    global_entropy /= coords.shape[0]
    
    perfectly_mixed_entropy = np.sum([1/len(np.unique(species))*np.log(1/len(np.unique(species))) for i in range(sample_count)])
    #print(perfectly_mixed_entropy)
    #print(global_entropy)

    return global_entropy/perfectly_mixed_entropy


def calc_coordination_number_index(coords, species, diameter = 0.0, kwargs_distance_matrix={}, verbose=0, _internal=0):

    if diameter < 0.0:
        print('Diameter must be non-negative.')
        return None
    
    # Get the unique species, and their number
    species_unique = np.unique(species)
    num_species = len(species_unique)
    if verbose != 0:
        print('Number of, and uniqe species: ', num_species, species_unique)
    # Initialize the array to store the average number of nearest neighbours
    coordination_numbers = np.zeros(num_species)
    # Loop over the species
    dist_matrix = calc_phi_ij(coords, coords, **kwargs_distance_matrix)
    particle_diameter = dist_matrix[np.triu_indices_from(dist_matrix,k=1)].min() if diameter == 0.0 else diameter
    #print('Particle diameter: ', particle_diameter)
    M = np.zeros(num_species)
    for i in range(num_species):
        # Get the distance matrix for the species in the form of a 2D array and of size (num_particles of this species, num_particles in total)
        dist_matrix_to_sort = dist_matrix[np.where(species==species_unique[i])[0],:]
        # Get the indices of the k nearest neighbours of each particle, this is the k nearest for each row.
        #indices_of_sorted_dist_mat = np.argpartition(dist_matrix_to_sort, kth=coords.shape[0], axis=1)
        indices_of_contact = np.where(dist_matrix_to_sort <= 1.1*particle_diameter)
        unique, count = np.unique(indices_of_contact[0], return_counts=True)
        #print('unique, count: ', unique, count)
        for j in range(len(unique)):
            n_nb = count[j]
            
            n_diff = len(np.where(species[np.argpartition(dist_matrix_to_sort[j,:], kth=n_nb+1)[1:n_nb+1]] != species_unique[i])[0])
            M[i] += n_diff/n_nb
    M *= num_species/coords.shape[0] # * num_species is, to achieve a  value of 1 for a perfectly mixed system (0 for a perfectly segregated system)
    
    return [np.sum(M)]


def calc_particle_scale_index(coords, species, diameter = 0.0, sample_method='random', sample_count=100, kwargs_distance_matrix={}, verbose=0):
    # Consists of two parts: the coordination number, and the lacey index. See respective functions for details.

    #####
    # Coordination number
    #####
    if diameter < 0.0:
        print('Diameter must be non-negative.')
        return None
    species_unique = np.unique(species)
    num_species = len(species_unique)
    if verbose != 0:
        print('Number of, and uniqe species: ', num_species, species_unique)
    dist_matrix = calc_phi_ij(coords, coords, **kwargs_distance_matrix)
    particle_diameter = dist_matrix[np.triu_indices_from(dist_matrix,k=1)].min() if diameter == 0.0 else diameter
    p = np.zeros((coords.shape[0], num_species))
    indices_of_contact = np.where(dist_matrix <= 1.1*particle_diameter)
    unique, count = np.unique(indices_of_contact[0], return_counts=True)
    C_n = np.zeros(coords.shape[0], dtype=int)
    C_n[unique] = count.astype(int)
    indices_of_contact_partitioned = np.argpartition(dist_matrix, kth=np.max(C_n), axis=1)
    unique_partitioned, count_partitioned = np.unique(indices_of_contact_partitioned, return_counts=True, axis=1)
    
    for i in range(coords.shape[0]):
        for s in range(num_species):
            C_nB = len(np.where(species[indices_of_contact_partitioned[i,1:C_n[i]+1]] == species_unique[s])[0]) # for loop over particles can not be avoided because C_n might be different for each particle and therefore we can not index indices_of_contact_partitioned with a 2D array 
            p[i,s] = C_nB/(C_n[i]+1)
    
    #####
    # Finish coordination number
    #####
            
    #####
    # Lacey index
    #####
    
    # Get the unique species, and their number
    species_unique = np.unique(species)
    num_species = len(species_unique)
    if verbose != 0:
        print('Number of, and uniqe species: ', num_species, species_unique)
    
    if sample_method == 'random':
        sample_origins_index = np.random.default_rng().choice(range(coords.shape[0]), sample_count)
    else:
        print('Could not determine sampling method for Lacey index. Currently supported sampling methods: \'random\'.')
    
    p_t = np.zeros(num_species)
    var_t = np.zeros(num_species)
    var_0 = np.zeros(num_species)
    var_r = np.zeros(num_species)

    p_t = np.sum(p, axis=0)/p.shape[0]
    for i in range(num_species):
        var_t[i] = np.sum(np.square(p[:,i] - p_t[i]))/(p.shape[0])
    var_0 = np.multiply(p_t, 1-p_t)
    var_r = var_0/p.shape[0]

    lacey_indices = np.zeros(num_species)
    
    
    lacey_indices = np.divide(var_t-var_0, var_r-var_0)
    
    #####
    # Finish Lacey index
    #####

    return lacey_indices
        