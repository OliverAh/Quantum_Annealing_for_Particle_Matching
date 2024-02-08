##########
#
# ToDo: Make matrices upper (or lower) triangular. As stated by dwave we should never submit both i,j and j,i
#       refs: - Youtube: https://youtu.be/jTDnGox0c9Y?list=PLPvKnT7dgEsujrrP7i_mgbkivBN3J6A8u&t=6221
#             - dwave docs: https://docs.dwavesys.com/docs/latest/handbook_reformulating.html#native-qpu-formulations-ising-and-qubo
#
#       --> objective is sum_i(a_i*x_i) + sum_i(sum_[j=i+1,...,n](b_ij*x_i*x_j))
#
########







import numpy as np
import scipy as sp


#Q
# converts triangular Q_array to triangular Q_dict
# Q_array MUST NOT be symmetric
def Q_convert_to_dict(Q_array):
    assert not ((0 == Q_array - np.transpose(Q_array)).all()), 'Q_array is symmetric'
    keys = []
    values = []
    Q_dict = {}
    Q_shape = np.shape(Q_array)
    for i in range(Q_shape[0]):
        for j in range(i, Q_shape[1]):
            elem = Q_array[i,j]
            if elem != 0:
                keys.append((i+1,j+1)) # examples start at 1, but 0 still works 
                values.append(elem)

    Q_dict = {keys[i]: values[i] for i in range(len(values))}
    return Q_dict

# converts symmetric Q_array to upper triangular Q_dict
# Q_array MUST be symmetric
def Q_convert_to_dict_sym(Q_array):
    assert ((0 == Q_array - np.transpose(Q_array)).all()), 'Q_array is not symmetric'
    keys = []
    values = []
    Q_dict = {}
    Q_shape = np.shape(Q_array)

    for i in range(Q_shape[0]):
        keys.append((i+1,i+1))
        values.append(Q_array[i,i])

    for i in range(Q_shape[0]):
        for j in range(i+1, Q_shape[1]):
            elem = Q_array[i,j]
            if elem != 0:
                keys.append((i+1,j+1)) # examples start at 1, but 0 still works 
                values.append(2*elem)

    Q_dict = {keys[i]: values[i] for i in range(len(values))}
    return Q_dict

def remove_large_distances(Q_old, distance_matrix, cut_abs):
    d_shape = np.shape(distance_matrix)
    ids_remove = np.where(distance_matrix > cut_abs) # shape is (2, num_particles) but is tuple, [0] rows, [1] cols
    #print(ids_remove[0])
    i_mats = ids_remove[0]*d_shape[0] + ids_remove[1]
    j_mats = ids_remove[1]*d_shape[0] + ids_remove[0]
    #print(i_mats)
    Q_new = Q_old
    #with np.printoptions(precision=3, suppress=True):
    #    print(Q_new)
    Q_new[i_mats, :] = 0
    Q_new[:, j_mats] = 0
    #with np.printoptions(precision=3, suppress=True):
    #    print(Q_new)
    return Q_new, i_mats, j_mats



#Q_1
def q_dist(distance_matrix):
    num_particles = np.shape(distance_matrix)[0]
    q_dist = np.zeros((num_particles*num_particles, num_particles*num_particles))

    for i in range(num_particles):
        for j in range(num_particles):
            j_mat = i*num_particles + j
            q_dist[i, j_mat] = distance_matrix[i,j]
    
    #print(q_dist)
    q_dist = np.matmul(q_dist.transpose(), q_dist) # normal symmetric matrix (before making upper triangular)
    #print(q_dist)
    temp_diag = np.diag(q_dist)
    q_dist = 2*np.triu(q_dist)
    q_dist -= np.diag(temp_diag)
    return q_dist

def q_dist_diag(distance_matrix):
    num_particles = np.shape(distance_matrix)[0]
    q_dist = np.zeros((num_particles*num_particles, num_particles*num_particles))

    for i in range(num_particles):
        for j in range(num_particles):
            j_mat = i*num_particles + j
            q_dist[j_mat, j_mat] = distance_matrix[i,j]
    return q_dist

def q_dist_diag_sparse(distance_matrix):
    num_particles = np.shape(distance_matrix)[0]
    #q_dist = sp.sparse.csr_array((num_particles*num_particles, num_particles*num_particles))
    data = np.zeros(num_particles*num_particles)
    row_indices = np.arange(num_particles*num_particles)
    col_indices = np.arange(num_particles*num_particles)
    for i in range(num_particles):
        for j in range(num_particles):
            j_mat = i*num_particles + j
            data[j_mat] = distance_matrix[i,j]
    q_dist = sp.sparse.csr_array((data, (row_indices.astype(int), col_indices.astype(int))))
    return q_dist

#Q_2
def q_part(num_particles):
    q_part = np.zeros((num_particles*num_particles, num_particles*num_particles))

    for i in range(0, num_particles*num_particles):
        q_part[i, i] = -1

    for i in range(0, num_particles):
        for j in range(0, num_particles):
            id_j = i*num_particles+j
            for k in range(j+1, num_particles):
                id_k = i*num_particles+k
                q_part[id_j, id_k] = 2 *2  # *2 is for making it upper triangular
    #print(q_part)
    return q_part

def q_part_sparse(num_particles):
    #q_part = np.zeros((num_particles*num_particles, num_particles*num_particles))
    #data = np.zeros(num_particles*num_particles)
    #row_indices = np.zeros(num_particles*num_particles)
    #col_indices = np.zeros(num_particles*num_particles)
    
    # number of nonzero entries is known beforehad from loop structure over i j k 
    #     -->  num_particles*num_particles*(num_particles-1)/2)
    #     which is much smaller that num_particles^4 
    #     --> sparse matrix is much smaller than dense matrix
    num_nonzeros = int(num_particles*num_particles*(num_particles-1)/2)
    data = np.zeros(num_nonzeros)
    row_indices = np.zeros(num_nonzeros)
    col_indices = np.zeros(num_nonzeros)


    #for i in range(0, num_particles*num_particles):
    #    q_part[i, i] = -1

    # off-diagonals, diagonals are subtracted later
    tmp_index = 0
    for i in range(0, num_particles):
        #print('   i = ', i)
        for j in range(0, num_particles):
            id_j = i*num_particles+j
            for k in range(j+1, num_particles):
                id_k = i*num_particles+k
                #q_part[id_j, id_k] = 2 *2  # *2 is for making it upper triangular
                row_indices[tmp_index] = id_j
                col_indices[tmp_index] = id_k
                data[tmp_index] = 2 *2
                tmp_index += 1
    #print(q_part)
    row_indices = row_indices.astype(int)
    col_indices = col_indices.astype(int)
    shape = (num_particles*num_particles, num_particles*num_particles)
    return sp.sparse.csr_array((data, (row_indices, col_indices)), shape=shape) - sp.sparse.eye((num_particles*num_particles), format='csr')

#Q_3
def q_pos(num_particles):
    q_pos = np.zeros((num_particles*num_particles, num_particles*num_particles))

    for j in range(0, num_particles*num_particles):
        q_pos[j, j] = -1

    for j in range(0, num_particles):
        for i in range(0, num_particles):
            id_i = i*num_particles+j
            for k in range(i+1, num_particles):
                id_k = k*num_particles+j
                q_pos[id_i, id_k] = 2 *2 # *2 is for making it upper triangular
    #print(Q_2)
    return q_pos

def q_pos_sparse(num_particles):
    #q_pos = np.zeros((num_particles*num_particles, num_particles*num_particles))
    
    # number of nonzero entries is known beforehad from loop structure over i j k 
    #     -->  num_particles*num_particles*(num_particles-1)/2)
    #     which is much smaller that num_particles^4 
    #     --> sparse matrix is much smaller than dense matrix
    num_nonzeros = int(num_particles*num_particles*(num_particles-1)/2)
    data = np.zeros(num_nonzeros)
    row_indices = np.zeros(num_nonzeros)
    col_indices = np.zeros(num_nonzeros)

    #for j in range(0, num_particles*num_particles):
    #    q_pos[j, j] = -1

    # off-diagonals, diagonals are subtracted later
    tmp_index = 0
    for j in range(0, num_particles):
        #print('   j = ', j)
        for i in range(0, num_particles):
            id_i = i*num_particles+j
            for k in range(i+1, num_particles):
                id_k = k*num_particles+j
                #q_pos[id_i, id_k] = 2 *2 # *2 is for making it upper triangular
                row_indices[tmp_index] = id_i
                col_indices[tmp_index] = id_k
                data[tmp_index] = 2 *2
                tmp_index += 1
    row_indices = row_indices.astype(int)
    col_indices = col_indices.astype(int)
    shape = (num_particles*num_particles, num_particles*num_particles)
    return sp.sparse.csr_array((data, (row_indices, col_indices)), shape=shape) - sp.sparse.eye((num_particles*num_particles), format='csr')