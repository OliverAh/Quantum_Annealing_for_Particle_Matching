import sys
import pathlib
#sys.path.append(str(pathlib.PurePath(pathlib.Path.cwd().parent)))
#sys.path.append(str(pathlib.PurePath(pathlib.Path.cwd().joinpath('Quantum_Annealing_for_Particle_Matching'))))
sys.path.append(str(pathlib.PurePath(pathlib.Path.cwd())))
for p in sys.path:
    print(p)
import os
import time
import h5py
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import gurobipy as gp

import dwave
import dwave.system
from src.particle_funcs import distance_matrix as distance_matrix
import src.leap_funcs.qubo.q_matrix as q_matrix
from src.leap_funcs import embedding_quality
from src.leap_funcs.qubo import filter_samples
from src._misc import compare_matrices

from src import h5py_funcs
from src.h5py_funcs import discoveries, init_custom_getstates, io, parameterstudy_using_info_file


# define particle arrangement
num_particles = 100

part_coords_n = np.zeros((num_particles,2))
part_coords_nm1 = np.zeros((num_particles,2))

for i in range(np.shape(part_coords_n)[0]):
    part_coords_n[i,:] = [0, i]
    part_coords_nm1[i,:] = [0.5*1, i]

fig_initial, axs_initial = plt.subplots(1,1)
axs_initial.scatter(part_coords_n[:,0], part_coords_n[:,1], label="n")
axs_initial.scatter(part_coords_nm1[:,0], part_coords_nm1[:,1], label="n-1")
fig_initial.legend()
#fig_initial.show()
fig_initial.savefig("./00_tests/08_02_test_gurobi_cluster/initial_config.png")
correct_sol = np.zeros(num_particles*num_particles)
for i in range(1, num_particles+1):
    correct_sol[(i-1)*num_particles + i -1] = 1.

print(correct_sol)

#compute matrices
distances = distance_matrix.calc_phi_ij(part_coords_n, part_coords_nm1)
#Q_dist_diag = q_matrix.q_dist_diag(distances)
Q_dist_diag_sparse = q_matrix.q_dist_diag_sparse(distances)
print('generated Q_dist')

#Q_part_sparse_slow = q_matrix.q_part_sparse(np.shape(distances)[0])
Q_part_sparse = q_matrix.q_part_sparse_fast(np.shape(distances)[0])
print('generated Q_part')
#assert compare_matrices.are_equal_sparse_matrices(Q_part_sparse_slow, Q_part_sparse)

#print(dir(Q_part_sparse))
#print(Q_part_sparse.data)
#print(Q_part_sparse.indices)
#print(Q_part_sparse.indptr)


#Q_pos_sparse_slow = q_matrix.q_pos_sparse(num_particles)
Q_pos_sparse = q_matrix.q_pos_sparse_fast(np.shape(distances)[0])
#assert compare_matrices.are_equal_sparse_matrices(Q_pos_sparse_slow, Q_pos_sparse)
print('generated Q_pos')
#print(Q_pos_sparse)

#Q_array = Q_dist_diag + Q_part + Q_pos
Q_array_sparse = Q_dist_diag_sparse + Q_part_sparse + Q_pos_sparse
print('generated Q_array')
#assert (Q_array_sparse.toarray() == Q_array).all(), 'Q_array_sparse is not equal to Q_array'

#with np.printoptions(threshold=np.inf, linewidth=np.inf, precision=2):
#    print(Q_pos_sparse.toarray())
#    print(Q_pos_sparse_slow.toarray())
#    print(Q_part_sparse.toarray())
#    print(Q_array_sparse.toarray())

#assert False, 'Stop here'

list_solvers_to_test = ['gurobi', 'scipy']
#list_solvers_to_test = ['scipy']

if 'gurobi' in list_solvers_to_test:
    # gurobi model
    gurobi_model = gp.Model()
    x = gurobi_model.addMVar(num_particles*num_particles, vtype='b', name='x')
    #Q = gurobi_model.addMVar(Q_array, name='Q')
    gurobi_model.setObjective(expr= x @ Q_array_sparse @ x, sense=gp.GRB.MINIMIZE)
    #gurobi_model.Params.Presolve = 0
    #gurobi_model.setParam('Presolve', 0)
    #gurobi_model.setParam('Threads', 1)
    print('    NumVars = ',gurobi_model.NumVars)
    gurobi_model.optimize()
    print('    NumVars = ',gurobi_model.NumVars)
    print(x.X)
    print ('Is solution correct? --> ', (x.X == correct_sol).all())
    print('Runtime = ', gurobi_model.Runtime)
    print()

if 'scipy' in list_solvers_to_test:
    # scipy model
    scipy_tic = time.time()
    scipy_row_ind, scipy_col_ind = sp.optimize.linear_sum_assignment(distances)
    scipy_toc = time.time()
    scipy_runtime = scipy_toc - scipy_tic
    scipy_x = np.zeros(num_particles*num_particles)
    for i in range(len(scipy_row_ind)):
        scipy_x[scipy_row_ind[i]*num_particles + scipy_col_ind[i]] = 1
    print(scipy_x)
    print()
    print ('Is solution correct? --> ', (scipy_x == correct_sol).all())
    print('Runtime = ', scipy_runtime)