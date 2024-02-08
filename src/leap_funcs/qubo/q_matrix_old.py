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


#Q
def Q_convert_to_dict(Q_array):
    keys = []
    values = []
    Q_dict = {}
    Q_shape = np.shape(Q_array)
    for i in range(Q_shape[0]):
        for j in range(Q_shape[1]):
            elem = Q_array[i,j]
            if elem != 0:
                keys.append((i+1,j+1)) # examples start at 1, but 0 still works 
                values.append(elem)
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
    
    #np.set_printoptions(precision=3)
    #print(q_dist)
    q_dist = np.matmul(q_dist.transpose(), q_dist)
    #print(q_dist)
    
    return q_dist


#Q_2
def q_part(num_particles):

    q_part = np.zeros((num_particles*num_particles, num_particles*num_particles))

    for i in range(1, num_particles+1):
        for j in range(1, num_particles+1):
            for k in range(1, num_particles+1):
                id_k = (i-1)*num_particles+k
                id_j = (i-1)*num_particles+j
                q_part[id_k-1, id_j-1] = 1 - 2*(j==k)
    #print(q_part)
    return q_part

#Q_3
def q_pos(num_particles):

    q_pos = np.zeros((num_particles*num_particles, num_particles*num_particles))

    for j in range(1, num_particles+1):
        for i in range(1, num_particles+1):
            for k in range(1, num_particles+1):
                id_k = (k-1)*num_particles+j
                id_i = (i-1)*num_particles+j
                q_pos[id_k-1, id_i-1] = 1 - 2*(i==k)
    #print(Q_2)
    return q_pos