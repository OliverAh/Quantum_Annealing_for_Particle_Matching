# %%
import sys
import pathlib
sys.path.append(str(pathlib.PurePath(pathlib.Path.cwd().parent)))

import os
import multiprocessing
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import dimod
import dwave
import dwave.system
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dwave.inspector
import dwave_networkx as dnx
import minorminer

from SALib.sample import sobol as salib_sample_sobol

from src.particle_funcs import distance_matrix as distance_matrix
from src.particle_funcs import io as particles_io
import src.leap_funcs.qubo.q_matrix as q_matrix

from src import leap_funcs as leap_funcs
from src.leap_funcs import embedding_quality
from src.leap_funcs.qubo import parameterstudy

from src import h5py_funcs
from src.h5py_funcs import inspections, discoveries, init_custom_getstates, io, parameterstudy_using_info_file

# %%
#folder_path_main = 'test_multi_param_emb_thr'
#info_file_name = 'info.h5'
#folder_embeddings = 'embeddings'
#embeddings_meta_file_name = '00_meta'
#folder_results = 'results'
#
#meta_file_name_path = os.path.join(folder_path_main, folder_embeddings, embeddings_meta_file_name)


# %%
def read_info_file(file_name_path='', infoset_name: str = ''):
    return h5py_funcs.io.read_info_from_hdf5_file(file_name_path=file_name_path, infoset_name=infoset_name)

def read_embeddings(reread_info_file = {}, folder_path_main=''):
    import ast
    embeddings = {}
    for key, file_names in reread_info_file['info']['embs_files']['file_names'].items():
        #print(key, file_names)
        for ind, file in enumerate(file_names['data']):
            file = file.decode('utf-8')
            file_name_path_emb = os.path.join(folder_path_main,'embeddings', file)
            emb = h5py_funcs.io.read_info_from_hdf5_file(
                file_name_path=file_name_path_emb, infoset_name='embedding_mm_01/embedding') 
            emb = {ast.literal_eval(var): list(qs['data']) for var, qs in emb.items()}
            embeddings.update({file: emb})
    return embeddings
def read_qubos(reread_info_file = {}, folder_path_main=''):
    import ast
    qubos = {}
    for key, qubo_read in reread_info_file['info']['qubos'].items():
        qubo = {ast.literal_eval(var): qs['data'] for var, qs in qubo_read.items()}
        qubos.update({key: qubo})
    return qubos

# %%


# %%
def _main_create_info_file(folder_path_main='', info_file_name=''):
    #num_particles = [5,7,10,13,15]
    num_particles = {'01': 5}
    
    #rocky_files = {'01': str(pathlib.Path(r"C:\zz_tmp_OAH\perf_sphere\qdem_sub_1\09_Particles_0025_CPU_4_GPU_0.rocky"))}
    rocky_files = {'01': str(pathlib.Path(r"C:\zz_tmp_OAH\perf_sphere\qdem_sub_1\01_Particles_0005_CPU_4_GPU_0.rocky"))}
    DEMs = {'num_particles': num_particles, 'rocky_files': rocky_files}
    
    num_nearest_neighbours = {'01': np.array([5])}
    
    qubos = {}
    for key, value in num_particles.items():
        print('creating QUBOs for {} particles and {} nearest neighbours'.format(value, num_nearest_neighbours[key]))
        folder_path = pathlib.Path(rocky_files[key]+r'.files')
        print('read DEM files from', folder_path)
        dem_data = particles_io.read_dem_data(folder_path=folder_path)

        time_values_for_snapshots = (57.5, 60)
        filenames_for_snapshots = []
        for key3, value3 in dem_data.items():
            if key3 not in ('num_particles', 'particle_index'):
                pass
            else:
                continue
            #print(value['attrs']['time_value'][0])
            if np.isclose(value3['attrs']['time_value'][0], time_values_for_snapshots).any():
                print('time index and value from DEM data:', value3['attrs']['time_index'], value3['attrs']['time_value'])
                filenames_for_snapshots.append(key3)
        particles_coords_names = dem_data[filenames_for_snapshots[0]]['Particles']['particles_position']['data'].dtype.names
        print('filenames for coords:', filenames_for_snapshots)

        part_coords_n = dem_data[filenames_for_snapshots[1]]['Particles']['particles_position']['data'].view((np.double, len(particles_coords_names)))
        part_coords_nm1 = dem_data[filenames_for_snapshots[0]]['Particles']['particles_position']['data'].view((np.double, len(particles_coords_names)))
        #part_coords_n

        distances = distance_matrix.calc_phi_ij(part_coords_n, part_coords_nm1)
        #distances

        Q_dist_diag = q_matrix.q_dist_diag(distances)
        Q_part = q_matrix.q_part(np.shape(distances)[0])
        Q_pos = q_matrix.q_pos(np.shape(distances)[0])
        Q_array = Q_dist_diag + Q_part + Q_pos
        #Q_dict = q_matrix.Q_convert_to_dict(Q_array)
        #del Q_dict
        Q_dict_new = q_matrix.Q_convert_to_dict_new_keys_iter(Q_array, value)
            
        for value2 in num_nearest_neighbours[key]: 
            print('Reduce Q-Matrix to {} nearest neighbours'.format(value2))
            Q_dict_to_store = Q_dict_new.copy()
            if value != value2:
                Q_dict_to_store = q_matrix.reduce_dict_to_nearest_neighbours(Q_dict_to_store, distances, value2)
            Q_dict_to_store = {str(_tmp1): _tmp2 for _tmp1, _tmp2 in Q_dict_to_store.items()}
            qubos.update({'{:0{w1}n}_{:0{w1}n}'.format(value, value2, w1=len(str(np.max(list(num_particles.values()))))): Q_dict_to_store}) 
    #print(qubos)


    embs_files = {}
    for key, value in num_nearest_neighbours.items():
        files_names = []
        for i in range(len(value)):
            files_names.append('emb_{:0{w1}n}_{:0{w1}n}_mm01.h5'.format(num_particles[key], value[i], w1=len(str(np.max(list(num_particles.values()))))))
        embs_files.update({key: np.array(files_names, dtype='S')})
    file_name_system = np.array('num_particles__nearest_neighbours__embedding_method', dtype='S')
    embs_files_names = {'file_name_system':file_name_system, 'file_names':embs_files}
    print(embs_files_names)
    del key, value

    
    names = ['num_particles', 'num_nearest_neighbours']
    study = np.zeros((np.sum([len(nei) for nei in num_nearest_neighbours.values()]), 2), dtype=np.int32)
    _tmp_counter = 0
    for key, value in num_particles.items():
        for value2 in num_nearest_neighbours[key]: 
            study[_tmp_counter,0] = value
            study[_tmp_counter,1] = value2
            _tmp_counter += 1
    del _tmp_counter, key, value, value2

    with open('../API_Token_Oliver_Dev.txt') as file:
        token = file.readline().rstrip()
    kwargs_dwavesampler = {'token' : token, 'region':'eu-central-1', 'architecture':'pegasus', 'name':'Advantage_system5.4'}
    sampler = DWaveSampler(**kwargs_dwavesampler)
    tmp_not_needed_as_a_variable = sampler.adjacency # required for sampler having all data needed for __getstate__, no idea why this is necessary
    #emb = minorminer.find_embedding(S=Q_dict_new, T=sampler.to_networkx_graph(), interactive=True, verbose=1)

    folder_name_embs = os.path.join(folder_path_main,'embeddings')
    file_name_emb = 'emb_{:0{w1}n}_{:0{w1}n}_mm01.h5'.format(num_particles['01'], num_nearest_neighbours['01'][0], w1=len(str(np.max(list(num_particles.values())))))
    emb = _find_embedding(Q_dict=Q_dict_new, folder_name_embs=folder_name_embs, 
                            file_name_emb=file_name_emb, kwargs_sampler=kwargs_dwavesampler,
                            rocky_analysis_path=rocky_files['01']+r'.files', layout_source_required=False)
    
    #for keyy, valuee in emb['mm_01'].items():
    #    print(keyy, type(valuee))
    #assert False
    emb = emb['mm_01']['embedding']




    names = ['annealing_time', 'programming_thermalization', 'readout_thermalization']
    salib_problem = {
    'num_vars': 3,
    'names': names,
    'bounds': [sampler.properties['annealing_time_range'],
               sampler.properties['programming_thermalization_range'],
               sampler.properties['readout_thermalization_range']]
    }
    #salib_problem['bounds'] = 3*[[0,1]]
    
    N = 2**8
    salib_sobol_samples = salib_sample_sobol.sample(salib_problem, N, calc_second_order=True, scramble=True)
    print('N ', N, ', D ', salib_problem['num_vars'])
    print('num samples: N(2D+2) = ', N*(2*salib_problem['num_vars']+2))
    print('shape', salib_sobol_samples.shape)
    print(salib_problem['names'])
    print(salib_sobol_samples)
    study = salib_sobol_samples

    num_qubits = len(set(inner for outer in emb.values() for inner in outer))
    print('num_qubits:', num_qubits)
    composite_fixed = FixedEmbeddingComposite(sampler, emb)
    est_time_s = np.zeros((salib_sobol_samples.shape[0], 1))
    for i in range(salib_sobol_samples.shape[0]):
        params_sampling = {'label' : 'superdupernice label',
                  'num_reads': 1000, 
                  'answer_mode': 'raw'}
        params_sampling.update({names[j]: salib_sobol_samples[i,j] for j in range(salib_sobol_samples.shape[1])})
        est_time_s[i] = composite_fixed.child.solver.estimate_qpu_access_time(num_qubits=num_qubits, **params_sampling)
        est_time_s[i] *= 1e-6
    print('estimated runtimes:', est_time_s)
    print('estimated overall runtime:', np.sum(est_time_s))    
    study = np.hstack((study, est_time_s))
    names.append('estimated_runtime')
    
    metadata_dict = {'Description': 'This file contains the information for the parameter study of the nearest neighbours',
                    'names': salib_problem['names']}
    problem_dict = {'names': salib_problem['names'], 'bounds': salib_problem['bounds'], 'num_vars': salib_problem['num_vars']}
    parametersets_array = study
    info_sets = {'DEMs':DEMs,
                 'nearest_neighbours':num_nearest_neighbours,
                 'qubos':qubos,
                'embs_files':embs_files_names}
    
    h5py_funcs.parameterstudy_using_info_file.prepare_info_file(
        metadata_dict = metadata_dict, 
        problem_dict = problem_dict, 
        parametersets_array = parametersets_array, 
        info_sets = info_sets, 
        info_set_name = 'info', 
        folder_path_name = folder_path_main, 
        info_file_name = info_file_name, 
        print_prefix='')
    
def _main_update_study_in_info_file(folder_path_main, info_file_name, old_info_file_name):
    num_particles = {'01': 5}
    rocky_files = {'01': str(pathlib.Path(r"C:\zz_tmp_OAH\perf_sphere\qdem_sub_1\01_Particles_0005_CPU_4_GPU_0.rocky"))}
    DEMs = {'num_particles': num_particles, 'rocky_files': rocky_files}
    
    num_nearest_neighbours = {'01': np.array([5])}

    qubos = {}
    for key, value in num_particles.items():
        print('creating QUBOs for {} particles and {} nearest neighbours'.format(value, num_nearest_neighbours[key]))
        folder_path = pathlib.Path(rocky_files[key]+r'.files')
        print('read DEM files from', folder_path)
        dem_data = particles_io.read_dem_data(folder_path=folder_path)

        time_values_for_snapshots = (57.5, 60)
        filenames_for_snapshots = []
        for key3, value3 in dem_data.items():
            if key3 not in ('num_particles', 'particle_index'):
                pass
            else:
                continue
            #print(value['attrs']['time_value'][0])
            if np.isclose(value3['attrs']['time_value'][0], time_values_for_snapshots).any():
                print('time index and value from DEM data:', value3['attrs']['time_index'], value3['attrs']['time_value'])
                filenames_for_snapshots.append(key3)
        particles_coords_names = dem_data[filenames_for_snapshots[0]]['Particles']['particles_position']['data'].dtype.names
        print('filenames for coords:', filenames_for_snapshots)

        part_coords_n = dem_data[filenames_for_snapshots[1]]['Particles']['particles_position']['data'].view((np.double, len(particles_coords_names)))
        part_coords_nm1 = dem_data[filenames_for_snapshots[0]]['Particles']['particles_position']['data'].view((np.double, len(particles_coords_names)))
        #part_coords_n

        distances = distance_matrix.calc_phi_ij(part_coords_n, part_coords_nm1)
        #distances

        Q_dist_diag = q_matrix.q_dist_diag(distances)
        Q_part = q_matrix.q_part(np.shape(distances)[0])
        Q_pos = q_matrix.q_pos(np.shape(distances)[0])
        Q_array = Q_dist_diag + Q_part + Q_pos
        #Q_dict = q_matrix.Q_convert_to_dict(Q_array)
        #del Q_dict
        Q_dict_new = q_matrix.Q_convert_to_dict_new_keys_iter(Q_array, value)
            
        for value2 in num_nearest_neighbours[key]: 
            print('Reduce Q-Matrix to {} nearest neighbours'.format(value2))
            Q_dict_to_store = Q_dict_new.copy()
            if value != value2:
                Q_dict_to_store = q_matrix.reduce_dict_to_nearest_neighbours(Q_dict_to_store, distances, value2)
            Q_dict_to_store = {str(_tmp1): _tmp2 for _tmp1, _tmp2 in Q_dict_to_store.items()}
            qubos.update({'{:0{w1}n}_{:0{w1}n}'.format(value, value2, w1=len(str(np.max(list(num_particles.values()))))): Q_dict_to_store}) 
    #print(qubos)

    embs_files = {}
    for key, value in num_nearest_neighbours.items():
        files_names = []
        for i in range(len(value)):
            files_names.append('emb_{:0{w1}n}_{:0{w1}n}_mm01.h5'.format(num_particles[key], value[i], w1=len(str(np.max(list(num_particles.values()))))))
        embs_files.update({key: np.array(files_names, dtype='S')})
    file_name_system = np.array('num_particles__nearest_neighbours__embedding_method', dtype='S')
    embs_files_names = {'file_name_system':file_name_system, 'file_names':embs_files}
    print(embs_files_names)
    del key, value



    reread_info_file = read_info_file(os.path.join(folder_path_main, old_info_file_name), infoset_name='')
    reread_embeddings = read_embeddings(reread_info_file=reread_info_file, folder_path_main=folder_path_main)
    print(reread_embeddings)
    emb = reread_embeddings['emb_5_5_mm01.h5'] # this is a dict in the form {(part i, pos j): [list of quibits]}

    with open('../API_Token_Oliver_Dev.txt') as file:
        token = file.readline().rstrip()
    kwargs_dwavesampler = {'token' : token, 'region':'eu-central-1', 'architecture':'pegasus', 'name':'Advantage_system5.4'}
    sampler = DWaveSampler(**kwargs_dwavesampler)
    tmp_not_needed_as_a_variable = sampler.adjacency # required for sampler having all data needed for __getstate__, no idea why this is necessary

    names = ['annealing_time', 'programming_thermalization', 'readout_thermalization']
    salib_problem = {
    'num_vars': 3,
    'names': names,
    'bounds': [sampler.properties['annealing_time_range'],
               sampler.properties['programming_thermalization_range'],
               sampler.properties['readout_thermalization_range']]
    }
    salib_problem['bounds'][0][1] *= .5
    salib_problem['bounds'][1][1] *= .5
    salib_problem['bounds'][2][1] *= .4
    
    N = 2**6
    salib_sobol_samples = salib_sample_sobol.sample(salib_problem, N, calc_second_order=True, scramble=True)
    print('N ', N, ', D ', salib_problem['num_vars'])
    print('num samples: N(2D+2) = ', N*(2*salib_problem['num_vars']+2))
    print('shape', salib_sobol_samples.shape)
    print(salib_problem['names'])
    print(salib_problem['bounds'])
    print(salib_sobol_samples)
    study = salib_sobol_samples

    num_qubits = len(set(inner for outer in emb.values() for inner in outer))
    print('num_qubits:', num_qubits)
    composite_fixed = FixedEmbeddingComposite(sampler, emb)
    est_time_s = np.zeros((salib_sobol_samples.shape[0], 1))
    for i in range(salib_sobol_samples.shape[0]):
        params_sampling = {'label' : 'superdupernice label',
                  'num_reads': 1000, 
                  'answer_mode': 'raw'}
        params_sampling.update({names[j]: salib_sobol_samples[i,j] for j in range(salib_sobol_samples.shape[1])})
        params_sampling.update({'flux_drift_compensation': False})
        est_time_s[i] = composite_fixed.child.solver.estimate_qpu_access_time(num_qubits=num_qubits, **params_sampling)
        est_time_s[i] *= 1e-6
    print('estimated runtimes:', est_time_s)
    print('estimated overall runtime in s:', np.sum(est_time_s))    
    print('estimated overall runtime in h:', np.sum(est_time_s)/3600)    
    study = np.hstack((study, est_time_s))
    print(study)

    #print(reread_info_file)

    metadata_dict = {'Description': 'This file contains the information for the parameter study of the nearest neighbours',
                    'names': salib_problem['names']}
    problem_dict = {'names': salib_problem['names'], 'bounds': salib_problem['bounds'], 'num_vars': salib_problem['num_vars']}
    parametersets_array = study
    info_sets = {'DEMs':DEMs,
                 'nearest_neighbours':num_nearest_neighbours,
                 'qubos':qubos,
                'embs_files':embs_files_names}
    
    h5py_funcs.parameterstudy_using_info_file.prepare_info_file(
        metadata_dict = metadata_dict, 
        problem_dict = problem_dict, 
        parametersets_array = parametersets_array, 
        info_sets = info_sets, 
        info_set_name = 'info', 
        folder_path_name = folder_path_main, 
        info_file_name = info_file_name, 
        print_prefix='')

# %%
def _find_embedding(Q_dict={}, folder_name_embs='', kwargs_sampler={}, file_name_emb='', rocky_analysis_path='', layout_source_required=False):

    if layout_source_required:
        folder_path = pathlib.Path(rocky_analysis_path)
        print(folder_path)
        dem_data = particles_io.read_dem_data(folder_path=folder_path)

        time_values_for_snapshots = (57.5, 60)
        filenames_for_snapshots = []
        num_particles = None
        particles_coords_names = None
        for key, value in dem_data.items():
            if key not in ('num_particles', 'particle_index'):
                pass
            else:
                continue
            #print(value['attrs']['time_value'][0])
            if np.isclose(value['attrs']['time_value'][0], time_values_for_snapshots).any():
                print(value['attrs']['time_index'], value['attrs']['time_value'])
                print(key)
                filenames_for_snapshots.append(key)
        num_particles = dem_data['num_particles'].astype(int)
        particles_coords_names = dem_data[filenames_for_snapshots[0]]['Particles']['particles_position']['data'].dtype.names
        print(filenames_for_snapshots, num_particles)
        print(particles_coords_names)
        print(type(num_particles))

        part_coords_n = dem_data[filenames_for_snapshots[1]]['Particles']['particles_position']['data'].view((np.double, len(particles_coords_names)))
        part_coords_nm1 = dem_data[filenames_for_snapshots[0]]['Particles']['particles_position']['data'].view((np.double, len(particles_coords_names)))
        #part_coords_n
        
        logic_vars = set(elem[0] for elem in list(Q_dict.keys()))
        layout_source = {elem:tuple(part_coords_n[elem[0]-1][:2]) for elem in logic_vars}
        #print(logic_vars)
        #print(layout_source)


    sampler = DWaveSampler(**kwargs_sampler)
    tmp_not_needed_as_a_variable = sampler.adjacency # required for sampler having all data needed for __getstate__, no idea why this is necessary
    sampler_graph = sampler.to_networkx_graph()

    sampler_dict = {}
    _tmp = sampler.client.config.dict().copy()
    _tmp.update({'token': 'removed_for_privacy'})
    sampler_dict.update({'client_reduced': _tmp.copy()})
    #sampler_dict.update({'solver_reduced': 
    _tmp = sampler.solver.data.copy()
    for key in list(_tmp['properties'].keys()):
        if key not in ('num_qubits', 'qubits', 'couplers', 'chip_id', 'tags', 'topology', 'category'):
            _tmp['properties'].pop(key)
    sampler_dict.update({'solver_reduced': _tmp.copy()})
    del _tmp

    

    kwargs_diffusion_candidates = {}
    #kwargs_diffusion_candidates_01 = {
    #    'tries':20, 
    #    'verbose':1,
    #    'layout':layout_source,
    #    #'vicinity':3,
    #    #'viscosity':,
    #    #'delta_t':,
    #    #'d_lim':,
    #    #'downscale':,
    #    #'keep_ratio':,
    #    #'expected_occupancy':
    #    }
    #kwargs_diffusion_candidates_02 = kwargs_diffusion_candidates_01.copy()
    #kwargs_diffusion_candidates_02.update({'tries':50})
    #kwargs_diffusion_candidates = {'mm_01': None, 'mm_02': None, 'em_01':kwargs_diffusion_candidates_01, 'em_02':kwargs_diffusion_candidates_02}
    kwargs_diffusion_candidates = {}
    kwargs_minorminer = {}
    kwargs_minorminer_mm_01 = {
    #        'max_no_improvement':250,
    #        #random_seed=None,
    #        #timeout=1000,
    #        #max_beta=None,
    #        'tries':250,
    #        #inner_rounds=None,
    #        'chainlength_patience':250,
    #        #max_fill=None,
    #        #threads=1,
    #        #return_overlap=False,
    #        #skip_initialization=False,
    #        #verbose=0,
    #        #interactive=False,
    #        #initial_chains=(),
    #        #fixed_chains=(),
    #        #restrict_chains=(),
    #        #suspend_chains=(),
    }
    #kwargs_minorminer_mm_02 = kwargs_minorminer_mm_01.copy()
    #kwargs_minorminer_mm_02.update({'max_no_improvement':500, 'tries':500, 'chainlength_patience':500})
    #kwargs_minorminer = {'mm_01':kwargs_minorminer_mm_01, 'mm_02':kwargs_minorminer_mm_02}
    kwargs_minorminer = {'mm_01':kwargs_minorminer_mm_01}

    kwargs_embera = {}
    #kwargs_embera.update({'em_01': None, 'em_02': None})

    timings_candidates = {}
    timings_embedding = {}
    #kwargs_wo_layout = kwargs_diffusion_candidates_01.copy()
    #kwargs_wo_layout.pop('layout')
    #timings_candidates.update({'mm_01':None, 'mm_02':None, 'em_01':None, 'em_02':None})



    embeddings = {}
    for key in kwargs_minorminer.keys():
        tic = time.time()
        embedding_mm = minorminer.find_embedding(S=Q_dict, T=sampler_graph, interactive=True, verbose=1, **kwargs_minorminer[key])
        toc = time.time()
        timings_embedding[key] = toc-tic
        embeddings[key] = {'embedding': embedding_mm, 'sampler': sampler_dict, 'qubo': Q_dict}
    #tic = time.time()
    #candidates_em_layout = embera.preprocess.diffusion_placer.find_candidates(Q_dict_edgelist, sampler_graph, **kwargs_diffusion_candidates)
    #toc = time.time()
    #timings_candidates['em_layout'] = toc-tic

    composites_fixed = {}
    for key, emb_sam in embeddings.items():
        sampler = DWaveSampler(**kwargs_sampler)
        tmp_not_needed_as_a_variable = sampler.adjacency # required for sampler having all data needed for __getstate__, no idea why this is necessary
        composites_fixed[key] = FixedEmbeddingComposite(sampler, emb_sam['embedding'])

    params_sampling = {'label' : 'superdupernice label',
              'annealing_time': 20, 
              'num_reads': 1000, 
              'answer_mode': 'raw', 
              'programming_thermalization': 1000, 
              'readout_thermalization': 0
              }

    for key, value in composites_fixed.items():
        emb = value.embedding
        #print(emb)
        num_qubits = len(set(inner for outer in emb.values() for inner in outer))
        estimate_time_ys = value.child.solver.estimate_qpu_access_time(num_qubits=num_qubits, **params_sampling)
        print('   {:22s}'.format(key), num_qubits, estimate_time_ys*1e-6)



    #folder_path = 'test_embeddings'
    #file_name = 'embeddings'
    file_name_path = os.path.join(folder_name_embs, file_name_emb)


    kwargs_file_writing_meta={'file_name_path': file_name_path,
                         'overwrite_data_in_file': False,
                         'track_order': True}

    meta_to_write = {
            'kwargs_diffusion_candidates': kwargs_diffusion_candidates, 
            'kwargs_minorminer': kwargs_minorminer,
            'kwargs_embera': kwargs_embera,
            'timings_candidates': timings_candidates,
            'timings_embedding': timings_embedding
            }
    h5py_funcs.io.write_to_hdf5_file(dict_data=meta_to_write, data_name='embedding', name_suffix='_00_meta', **kwargs_file_writing_meta)

    for key, emb in embeddings.items():
        #print(emb)
        emb_to_write = emb
        meta_to_write_emb = {key_meta: value_meta[key] for key_meta, value_meta in meta_to_write.items() if key in value_meta}
        kwargs_file_writing_embs = kwargs_file_writing_meta.copy()
        #kwargs_file_writing_embs.update({'file_name_path': os.path.join(folder_path, key+'.h5')})
        h5py_funcs.io.write_to_hdf5_file(dict_data=emb_to_write, data_name='embedding', name_suffix='_'+key, **kwargs_file_writing_embs)
        h5py_funcs.io.write_to_hdf5_file(dict_data=meta_to_write_emb, data_name='embedding', name_suffix='_'+key+'_meta', **kwargs_file_writing_embs)

    return embeddings

# %%
def _main_find_embeddings(reread_info_file={}, folder_path_main=''):
    import ast
    with open('../API_Token_Oliver_Dev.txt') as file:
        token = file.readline().rstrip()
        architecture = file.readline().rstrip()
    kwargs_sampler = {'token' : token, 'architecture':'pegasus', 'region':'eu-central-1'}
    max_num_particles = np.max([reread_info_file['info']['DEMs']['num_particles'][key]['data'] for key in reread_info_file['info']['DEMs']['num_particles'].keys()]) 
    for key, dem in reread_info_file['info']['DEMs']['rocky_files'].items():
        num_particles = reread_info_file['info']['DEMs']['num_particles'][key]['data']
        rocky_files_path = dem['data'].decode('utf-8')+r'.files'
        print('number of particles:', num_particles, 'DEM folder:', rocky_files_path)
        for ind, nearest_neighbours in enumerate(reread_info_file['info']['nearest_neighbours'][key]['data']):
            file_name_emb = reread_info_file['info']['embs_files']['file_names'][key]['data'][ind].decode('utf-8')
            print('number of nearest neighbours:', nearest_neighbours, 'file_name_for_embedding', file_name_emb)
            qubos_key = '{:0{w1}n}_{:0{w1}n}'.format(num_particles, nearest_neighbours, w1=len(str(max_num_particles)))
            Q_dict = reread_info_file['info']['qubos'][qubos_key]
            Q_dict = {ast.literal_eval(var): qs['data'] for var, qs in Q_dict.items()}
            folder_name_embs = os.path.join(folder_path_main,'embeddings')
            #print(Q_dict)
            #assert False
            _find_embedding(Q_dict=Q_dict, folder_name_embs=folder_name_embs, 
                            file_name_emb=file_name_emb, kwargs_sampler=kwargs_sampler,
                            rocky_analysis_path=rocky_files_path, layout_source_required=False)
    

# %%
def overloaded_submitter_work(self, problem, verbose=0, print_prefix=''):
    identifier = problem['identifier']
    print(print_prefix + 'start working on problem {}'.format(identifier))
    try:
        embedding=problem['embedding']
        sampler = DWaveSampler(**self.solver)
        tmp_not_needed_as_a_variable = sampler.adjacency # required for sampler having all data needed for __getstate__, no idea why this is necessary
        composite = FixedEmbeddingComposite(child_sampler=sampler, embedding=embedding)
        num_qubits = len(set(inner for outer in embedding.values() for inner in outer))
        estimate_time_ys = composite.child.solver.estimate_qpu_access_time(num_qubits=num_qubits, **problem['kwargs_sampling'])
        _tmp = np.ceil(1.25*estimate_time_ys*1e-6).astype(int)
        num_reads = problem['kwargs_sampling']['num_reads']
        if _tmp <= 1:
            print(print_prefix + 'problem {} will take can be sampled with a single run'.format(identifier))
            
            num_reads = [num_reads]
        elif _tmp > 1:
            print(print_prefix + 'problem {} will take too long ({}) to solve'.format(identifier, estimate_time_ys))
            problem['num_runs']
            n, r = divmod(num_reads, _tmp)
            num_reads = [n]*_tmp
            num_reads[0] += r
        print(print_prefix + 'problem {} will be solved {} times per run'.format(identifier, len(num_reads)))
        dict_data = {'set_identifier': identifier, 'composite': composite.__getstate__().copy()}
        self._lock_info_file.acquire()
        h5py_funcs.parameterstudy_using_info_file.update_timestamp_in_info_file(file_name_path=os.path.join(self.folder_path_main, self.info_file_name), 
                        info_set='info', set_identifier=identifier, name='start', 
                        timestamp=h5py_funcs.parameterstudy_using_info_file._current_datetime_as_string())
        self._lock_info_file.release()

        samples = {}
        is_not_first_runread = False
        num_runs = problem['num_runs']
        for run in range(num_runs):
            print(print_prefix + 'start run {} for problem {}'.format(run, identifier))
            for id_read, read in enumerate(num_reads):
                print(print_prefix + '  start samplings {} for problem {}'.format(read, identifier))
                kwargs_sample_qubo = problem['kwargs_sampling'].copy()
                kwargs_sample_qubo.update({'num_reads': read})
                kwargs_sample_qubo['label'] += '_' + str(run) + '_' + str(id_read)
                _ans = composite.sample_qubo(**kwargs_sample_qubo)
                print('sampled for problem {}'.format(identifier))
                
                #_answer = _ans.__dict__.copy()
                _ans.resolve()
                _answer = _ans.__dict__.copy()
                #for key, value in _answer.items():
                #    print(key, value)
                #    hash(key)
                #    if key in ('_record', '_variables', '_info'):
                #        pass
                #        #data = value.tolist()
                #        #print(data)
                #        #data = [tuple(elem) for elem in data]
                #        #print(data)
                #        #hash(data)
                #        #
                #        #dd = dict(type='array', data=value.tolist(), data_type=value.dtype.name, shape=value.shape)
                #        #hash(data)
                #        #hash(value.tolist())
                #        #hash(value.tolist())
                #        #hash(dd)
                #    else:
                #        hash(value)
                ##for key, value in _answer['_info'].items():
                ##    print(key, value)
                ##    hash(key)
                ##    hash(value)
                _record = _answer['_record']
                _variables = _answer['_variables']
                _info = _answer['_info']
                _info['warnings'] = str(_info['warnings'])
                #_answer.pop('_record')
                #_answer.pop('_variables')
                #_answer.pop('_info')
                _ans = _answer
                _ans['_info']['warnings'] = str(_info['warnings'])
                #_ans.pop('_record')
                #_ans.pop('_variables')
                #_ans.pop('_info')
                _ans.pop('_vartype')
                #_ans = _ans.to_serializable(pack_samples=False)
                print('sampled for problem {}'.format(identifier))

                #if is_not_first_runread:
                #    _ans.pop('embedding_context')
                #else:
                #    pass
                samples.update({'{:0{width}n}_{:0{width2}n}_{:0{width2}n}'.format(run,read,id_read,width=len(str(100)),width2=len(str(1000))) : _ans})
                is_not_first_runread = True
                print('finished samples sampling for {}'.format(identifier))
            print(print_prefix + 'finished run {} for problem {}'.format(run, identifier))

        self._lock_info_file.acquire()
        h5py_funcs.parameterstudy_using_info_file.update_timestamp_in_info_file(file_name_path=os.path.join(self.folder_path_main, self.info_file_name), 
                        info_set='info', set_identifier=identifier, name='finish', 
                        timestamp=h5py_funcs.parameterstudy_using_info_file._current_datetime_as_string())
        self._lock_info_file.release()

        dict_data.update({'sampleset': samples})

        kwargs_write_answer = {'file_name_path': os.path.join(self.folder_path_main, 'samples', identifier+'.h5'),
                                'dict_data': dict_data.copy(),
                                'data_name': 'custom',
                                'name_suffix': '', 
                                'overwrite_data_in_file': False,
                                'track_order': True}
        print(print_prefix + 'succesfully retrieved answer for problem {}', identifier)
        return kwargs_write_answer
    except Exception as e:
        print(print_prefix + 'error working on problem {}'.format(identifier))
        #print(print_prefix + e.__str__())
        #print(e.__traceback__.tb_frame)
        leap_funcs.qubo.parameterstudy.traceback.print_exception(type(e), e, e.__traceback__)
        raise


# %%
def overloaded_writer_work(self, answer, verbose=0, print_prefix=''):
    try:
        print(print_prefix + 'start writing answer for problem {}', answer['dict_data']['set_identifier'])
        h5py_funcs.io.write_to_hdf5_file(**answer)
        print(print_prefix + 'finished writing answer for problem {}', answer['dict_data']['set_identifier'])
        return
    except Exception as e:
        #print(print_prefix + 'error writing answer for problem {}'.format(answer['dict_data']['set_identifier']))
        #print(print_prefix + e.__str__())
        leap_funcs.qubo.parameterstudy.traceback.print_exception(type(e), e, e.__traceback__)
        raise

# %%
def child_process_target(**kwargs):
    cpn = multiprocessing.current_process().name
    print(kwargs['target']['print_prefix'] + cpn, 'started with inputs', kwargs['target'])
    #answer = overloaded_submitter_work(*kwargs['submitter']['args'], **kwargs['submitter']['kwargs'])
    #overloaded_writer_work(*((answer,) + kwargs['writer']['args']), **kwargs['writer']['kwargs'])

    st = leap_funcs.qubo.parameterstudy.Multithread_Variationstudy(
        num_threads_submitters=8, num_threads_writers=2
    )
    st.submitter_work = overloaded_submitter_work
    st.writer_work = overloaded_writer_work
    st.folder_path_main = kwargs['target']['folder_path_main']
    st.info_file_name = kwargs['target']['info_file_name']
    st.problems = kwargs['submitter']['args']
    st.solver.update(**kwargs['target']['kwargs_dwavesampler'])
    st.start_execution(verbose=0)


# %%
def _main_create_chunks(arg={}, num_chunks=0):
    splitter = np.array_split(list(arg.keys()), num_chunks)
    chunks = [[]]*num_chunks
    for i in range(num_chunks):
        chunks[i] = [arg[key] for key in splitter[i]]
    #chunks = [arg[splitter[j][i]] for j in range(num_chunks) for i in range(len(splitter[j])) ]
    return chunks

# %%
def main():
    folder_path_main = os.path.join(os.getcwd(), '01_out', 'sub_4_2')
    info_file_name = 'study_params_sub4_2.h5'
    
    reread_info_file = read_info_file(os.path.join(folder_path_main, info_file_name), infoset_name='')
    reread_embeddings = read_embeddings(reread_info_file=reread_info_file, folder_path_main=folder_path_main)
    reread_qubos = read_qubos(reread_info_file=reread_info_file, folder_path_main=folder_path_main)
    sampler_params = {}
    #for i, id in enumerate(reread_info_file['info']['study']['data']['identifiers'][:len(reread_embeddings)]):
    
    st = reread_info_file['info']['study']['data'][1]
    print(st, st.dtype.names)
    print(st['sets'], st['sets'].dtype.names)
    names = [name for name in st['sets'].dtype.names if name != 'estimated_runtime']
    print(names)

    for study in reread_info_file['info']['study']['data']:
        names = [name for name in study['sets'].dtype.names if name != 'estimated_runtime']
        num_particles = int(reread_info_file['info']['DEMs']['num_particles']['01']['data'])
        num_nearest_neighbours = int(int(reread_info_file['info']['nearest_neighbours']['01']['data']))
        
        id = study['identifiers']
        print('num_particles:', num_particles, 'nearest_neighbours:', num_nearest_neighbours)
        max_num_particles = np.max([reread_info_file['info']['DEMs']['num_particles'][key]['data'] for key in reread_info_file['info']['DEMs']['num_particles'].keys()])
        print('max_num_particles:', max_num_particles)
        key_study = '{:0{w}n}_{:0{w}n}'.format(num_particles, num_nearest_neighbours, w=len(str(max_num_particles)))
        qubos_key = key_study
        embs_key = 'emb_'+key_study+'_mm01.h5'
        _qubo = reread_qubos[qubos_key] # just to make sure that the qubo is there
        _emb = reread_embeddings[embs_key] # just to make sure that the embedding is there
        
        #print(reread_embeddings[embs_key])
        #print(_qubo)
        sp_update = {id.decode('utf-8'): 
                        {'identifier': id.decode('utf-8'),
                         'embedding': _emb,
                         'num_runs': 10,
                         'kwargs_sampling': {'Q':_qubo, 'num_reads':1000, 'label':id.decode('utf-8')}}
                         }
        sp_update[id.decode('utf-8')]['kwargs_sampling'].update({name: study['sets'][name] for name in names})
        sp_update[id.decode('utf-8')]['kwargs_sampling'].update({'flux_drift_compensation': True})
        
        if reread_info_file['info']['time_history'][id.decode('utf-8')]['attrs']['finished'] == False:
            print(id.decode('utf-8'), 'would be executed')
            sampler_params.update(sp_update)
        #print(sampler_params)
    print(len(list(sampler_params.keys())), 'psets will be executed')
    #sys.exit()
    
    iterations = len(sampler_params)
    chunk_size = 25
    num_chunks = np.ceil(iterations/chunk_size).astype(int)
    chunks = _main_create_chunks(sampler_params, num_chunks)
    #print(chunks)
    #assert False
    multiprocessing.set_start_method('spawn')
    print('num_chunks =', num_chunks)
    print(__name__)
    
    with open('../API_Token_MBD_qdem.txt', mode='rt') as file:
    #with open('../API_Token_Oliver_Dev.txt', mode='rt') as file:
        token = file.readline().rstrip()
        #kwargs_dwavesampler = {'token': token, 'architecture': 'pegasus', 'region': 'eu-central-1'}

    process_counter = 0
    process_list = []
    for chunk_id, chunk in enumerate(chunks):
        print('chunk', chunk_id+1, 'of', num_chunks)
        #print(chunk)
        #p = multiprocessing.Process(target=test_function.f_test)
        kwargs_dwavesampler = {'token' : token, 'region':'eu-central-1', 'architecture':'pegasus', 'name':'Advantage_system5.4'}
        kwargs_target = {'print_prefix': ' ', 'kwargs_dwavesampler': kwargs_dwavesampler, 'folder_path_main': folder_path_main, 'info_file_name': info_file_name}
        inputs_submitter = {'args': chunk, 'kwargs':{'print_prefix': ' '}}
        inputs_writer = {'args': (), 'kwargs':{'print_prefix': ' '}}
        inputs_target = {'args': (), 'kwargs':{'target': kwargs_target, 'submitter': inputs_submitter, 'writer': inputs_writer}}

        
        p = multiprocessing.Process(target=child_process_target, args=inputs_target['args'], kwargs=inputs_target['kwargs'])
        process_list.append(p)
        process_counter += 1
        if (process_counter >= 8) or (chunk_id == len(chunks)-1):
            for pp in process_list:
                pp.start()
            for pp in process_list:
                pp.join()
                pp.close()
            process_counter = 0
            process_list = []
        #p.start()
        #p.join()
        #p.close()
    
    
    
    
    
    return sampler_params

# %%
def main_reset_notfinished_runs_info_file():
    folder_path_main = os.path.join(os.getcwd(), '01_out', 'sub_4_2')
    info_file_name = 'study_params_sub4_2.h5'
    info_file_name_path = os.path.join(folder_path_main, info_file_name)

    dict_info_read = h5py_funcs.inspections.read_info_file_to_dict(info_file_name_path=info_file_name_path, infoset_name = 'info')
    array_identifiers, started_psets, finished_psets = h5py_funcs.inspections.extract_identifiers(dict_info_read = dict_info_read)
    #print(dict_info_read['study']['data']['sets']['estimated_runtime'])
    #print(started_psets)
    #print(finished_psets)
    print('{p1}/{p2} p_sets have been started, \n {p3}/{p2} p_sets have been finished'.format(p1=len(started_psets), p2=array_identifiers.shape[0], p3=len(finished_psets)))
    print(f'estimated runtime all [s]: {np.sum(dict_info_read['study']['data']['sets']['estimated_runtime'])}')
    print(f'estimated runtime finished [s]: {np.sum(dict_info_read['study']['data']['sets']['estimated_runtime'][np.isin(dict_info_read['study']['data']['identifiers'], finished_psets)])}')

    with h5py.File(info_file_name_path,'r+') as f:
        for obj in f['info']['time_history'].values():
            if obj.attrs['finished']==False:# or obj.attrs['finished']==True:
                print(type(obj), obj, obj.attrs, obj[()])

                obj[1] = np.array(h5py_funcs.parameterstudy_using_info_file._current_datetime_as_string(-1e6))
                obj[2] = np.array(h5py_funcs.parameterstudy_using_info_file._current_datetime_as_string(-1e6))

                obj.attrs['finished'] = False

# %%
if __name__ == '__main__':
    print('This is the main process')
    #main_reset_notfinished_runs_info_file()
    main()


# %%



