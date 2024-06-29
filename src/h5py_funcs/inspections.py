import numpy as np
import matplotlib.pyplot as plt

import sys
import pathlib
sys.path.append(str(pathlib.PurePath(pathlib.Path.cwd().parent)))

import os
import multiprocessing
import multiprocessing.pool
import time
import h5py
import numpy as np
import matplotlib.cm as cm
import matplotlib.axes as am
import matplotlib.pyplot as plt
import plotly.express as px
#%matplotlib widget
import networkx as nx
import pandas as pd

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
from src.h5py_funcs import discoveries, init_custom_getstates, io, parameterstudy_using_info_file

'''
ToDo: move inspections from 24_inspect_run.ipynb to this file 
'''

def read_info_file_to_dict(info_file_name_path:str='', infoset_name:str='') -> dict:
    if info_file_name_path=='' or infoset_name=='':
        raise ValueError('All inputs required')
    else:
        return h5py_funcs.io.read_info_from_hdf5_file(file_name_path = info_file_name_path, infoset_name = infoset_name)

def extract_identifiers(dict_info_read:dict={}) -> list[np.ndarray,list,list]:
    if dict_info_read=={}:
        raise ValueError('info dict is required')
    else:
        array_identifiers = dict_info_read['study']['data']['identifiers']
        started_psets = []
        finished_psets = []
        for p_set_id in array_identifiers:
            t_hist_pset_data = dict_info_read['time_history'][p_set_id.decode('utf-8')]['data']
            yyyy_start = t_hist_pset_data[1].decode('utf-8')[:4] # year is sufficient in this case, because default date is in year 1910
            yyyy_finish = t_hist_pset_data[2].decode('utf-8')[:4] # year is sufficient in this case, because default date is in year 1910
            if yyyy_start == '2024':
                started_psets.append(p_set_id)
            if yyyy_finish == '2024':
                finished_psets.append(p_set_id)
        return array_identifiers, started_psets, finished_psets
        
def read_answers_to_dict(samples_folder_name_path:pathlib.Path=None, array_identifiers:np.ndarray=None, num_threads:int=2) -> dict:
    if samples_folder_name_path==None:
        raise ValueError('samples_folder_name_path is required')
    elif isinstance(samples_folder_name_path, str):
        raise TypeError('samples_folder_name_path is of type str but must be of type pathlib.Path')
    elif not isinstance(array_identifiers,np.ndarray):
        raise ValueError('array_identifiers is required, you may obtain it from .extract_identifiers(...)')
    list_files_in_samples_dir = os.listdir(samples_folder_name_path)
    def _read_answs_to_dict(p_set_id):
        global it, dict_for_df
        it += 1
        p_set_id_dec = p_set_id.decode('utf-8')
        set_in_dir = p_set_id_dec+'.h5' in list_files_in_samples_dir
        print(f'{it}/{array_identifiers.shape[0]} file {p_set_id_dec+'.h5'} in dir: {set_in_dir}\r', end='')
        if set_in_dir:
            p_set_file_name_path = pathlib.Path.joinpath(samples_folder_name_path, p_set_id_dec+'.h5')
            dict_for_df.update({p_set_id_dec: h5py_funcs.io.read_info_from_hdf5_file(file_name_path=p_set_file_name_path, driver='core')})
        return
    # apparently reading hdf5 files is not io bound in this case, so 1 thread does as well as more
    #for num_threads in (50,25,15,10,5):
    #for num_threads in (1,2,3,4,5):
    for _num_threads in (num_threads,):
        global it, dict_for_df
        it = 0
        dict_for_df = {}
        tic = time.time()
        tp = multiprocessing.pool.ThreadPool(_num_threads)
        tp.map(_read_answs_to_dict, array_identifiers)
        print()
        toc = time.time()
        #print(f'\n{_num_threads} took [s] {toc-tic}')
    #        #160/2048 file zz_7106551505.h5 in dir: True
    #        #1 took [s] 75.3208556175232
    #        #310/2048 file zz_7106551505.h5 in dir: True
    #        #2 took [s] 66.5232310295105
    #        #460/2048 file zz_1523292874.h5 in dir: True
    #        #3 took [s] 68.72280383110046
    #        #610/2048 file zz_7106551505.h5 in dir: True
    #        #4 took [s] 70.92504715919495
    #        #760/2048 file zz_7106551505.h5 in dir: True
    #        #5 took [s] 76.96953821182251
    return dict_for_df

def extract_success_dict(dict_for_df:dict=None, exact_sols:np.recarray=None, n_samples_to_compare:int=0, n_exact_sols_to_compare:int=0):
    if dict_for_df==None or n_samples_to_compare==0 or n_exact_sols_to_compare==0 or not isinstance(exact_sols,np.recarray):
        raise ValueError('all inputs are required')

    def _group_subs_per_run(samplesets_names:list[str]):
        names_split = [_tmp.split('_') for _tmp in samplesets_names]
        #assert all([len(n)==2 for n in names_split]), 'names_split must be list of lists, where each inner list is of len 2 exactly, because submissions are named by run_submission'
        names_runs_unique = [key for key in dict.fromkeys(s[0] for s in names_split).keys()] # contains same elements as "set(s[0] for s in names_split)", but additionally preserves order
        #print(names_runs_unique)
        names_runs_unique_iter = iter(names_runs_unique)
        names_subs_per_run = []
        _tmp_str = next(names_runs_unique_iter)
        _tmp_list = []
        for name_split in names_split:
            #print(name_split)
            #print(_tmp_str)
            if name_split[0] != _tmp_str:
                names_subs_per_run.append(_tmp_list)
                _tmp_list = [name_split[1:]]
                _tmp_str = next(names_runs_unique_iter)    
            else:
                _tmp_list.append(name_split[1:])
        names_subs_per_run.append(_tmp_list)
        return names_runs_unique, names_subs_per_run    

    def _evaluate_success_rate(exact_sols:dimod.sampleset.SampleSet=None, sample_sols:dimod.sampleset.SampleSet=None, n_exact:int=1, n_samples:int=1, is_print_sols:bool=False, is_print_meta:bool=False, print_prefix:str=' '):
        samplesets_names = list(sample_sols.keys())
        names_runs, names_subs_per_run = _group_subs_per_run(samplesets_names=samplesets_names)
        num_runs = len(names_runs)
        num_subs_per_run = [len(n) for n in names_subs_per_run]
        num_samples = 0
        num_samples_per_run = []
        num_samples_per_sub_per_run = []
        num_runs_is_found_best = 0
        num_samples_is_found_best = 0
        num_samples_is_found_best_per_run = []
        is_found_best_per_run = []
        num_matched = 0
        num_matched_per_run = []
        num_matched_per_sub_per_run = []
        num_samples_matched = 0
        num_samples_matched_per_run = []
        num_samples_matched_per_sub_per_run = []

        if is_print_meta:
            print(print_prefix, 'Sampleset keys:', samplesets_names)
            print(print_prefix, 'Names runs:', names_runs)
            print(print_prefix, 'Names subs per run:', names_subs_per_run)
            print(print_prefix, 'Number of runs:', num_runs)
            print(print_prefix, 'Number of submissions per run:', num_subs_per_run)

        return_dict = {}
        return_dict['num_runs_is_found_best'] = num_runs_is_found_best
        return_dict['num_samples_is_found_best'] = num_samples_is_found_best
        return_dict['num_samples_is_found_best_per_run'] = num_samples_is_found_best_per_run
        return_dict['is_found_best_per_run'] = is_found_best_per_run
        return_dict['num_samples'] = 0
        return_dict['num_runs'] = num_runs
        return_dict['num_subs_per_run'] = num_subs_per_run
        return_dict['num_samples'] = num_samples
        return_dict['num_samples_per_run'] = num_samples_per_run
        return_dict['num_samples_per_sub_per_run'] = num_samples_per_sub_per_run
        return_dict['num_matched'] = num_matched
        return_dict['num_matched_per_run'] = num_matched_per_run
        return_dict['num_matched_per_sub_per_run'] = num_matched_per_sub_per_run
        return_dict['num_samples_matched'] = num_samples_matched
        return_dict['num_samples_matched_per_run'] = num_samples_matched_per_run
        return_dict['num_samples_matched_per_sub_per_run'] = num_samples_matched_per_sub_per_run
        return_dict['submissions'] = {}
        for _i, name_run in enumerate(names_runs):
            num_samples_per_run.append(0)
            num_samples_per_sub_per_run.append([])
            is_found_best_per_run.append(False)
            num_matched_per_run.append(0)
            num_matched_per_sub_per_run.append([])
            num_samples_matched_per_run.append(0)
            num_samples_matched_per_sub_per_run.append([])
            num_samples_is_found_best_per_run.append(0)
            for name_sub in names_subs_per_run[_i]:
                if isinstance(name_sub, list):
                    name = name_run
                    for n in name_sub:
                        name += '_' + n
                elif isinstance(name_sub, str):
                    name = name_run + '_' + name_sub
                else:
                    raise ValueError(f'Could not create key of sampleset from name_sub={name_sub}')

                samples = sample_sols[name]['_record']['data']
                samples.sort(order='energy')
                if is_print_sols:
                    print(print_prefix, 'Sampled solutions')
                    print(samples[:n_samples])
                    print(print_prefix, 'Exact solutions')
                    print(exact_sols[:n_exact])

                #num_samples += samples.shape[0]
                _num_samples_sub = np.sum(samples['num_occurrences'])
                num_samples +=  _num_samples_sub
                num_samples_per_run[-1] += _num_samples_sub
                num_samples_per_sub_per_run[-1].append(_num_samples_sub)
                return_dict['num_samples'] = num_samples
                return_dict['num_samples_per_run'] = num_samples_per_run
                return_dict['num_samples_per_sub_per_run'] = num_samples_per_sub_per_run
                return_dict['submissions'][name] = {}
                return_dict['submissions'][name]['num_samples'] = _num_samples_sub
                return_dict['submissions'][name]['is_found_best'] = np.array_equal(samples['sample'][0], exact_sols['sample'][0])
                is_found_best_per_run[-1] = is_found_best_per_run[-1] or return_dict['submissions'][name]['is_found_best']
                if return_dict['submissions'][name]['is_found_best']:
                    num_samples_is_found_best += samples['num_occurrences'][0]
                    num_samples_is_found_best_per_run[-1] += (samples['num_occurrences'][0])
                return_dict['submissions'][name]['num_matched'] = 0
                return_dict['submissions'][name]['matches_sample_exact'] = []
                num_matched_per_sub_per_run[-1].append(0)
                num_samples_matched_per_sub_per_run[-1].append(0)
                for i_s in range(n_samples):
                    for i_e in range(n_exact):
                        is_contained = np.array_equal(samples['sample'][i_s], exact_sols['sample'][i_e])
                        if is_print_meta:
                            print('matched samp exac:', i_s, i_e, is_contained)
                        if is_contained:
                            num_matched += 1
                            num_matched_per_run[-1] += 1
                            num_matched_per_sub_per_run[-1][-1] += 1
                            num_samples_matched += samples['num_occurrences'][i_s]
                            num_samples_matched_per_run[-1] += samples['num_occurrences'][i_s]
                            num_samples_matched_per_sub_per_run[-1][-1] += samples['num_occurrences'][i_s]

                            return_dict['submissions'][name]['matches_sample_exact'].append((i_s, i_e))
                return_dict['submissions'][name]['num_matched'] = num_matched_per_sub_per_run[-1][-1].real
                return_dict['num_matched'] = num_matched
                return_dict['num_matched_per_run'] = num_matched_per_run
                return_dict['num_matched_per_sub_per_run'] = num_matched_per_sub_per_run
                return_dict['num_samples_matched'] = num_samples_matched
                return_dict['num_samples_matched_per_run'] = num_samples_matched_per_run
                return_dict['num_samples_matched_per_sub_per_run'] = num_samples_matched_per_sub_per_run
        num_runs_is_found_best = np.sum(is_found_best_per_run)
        return_dict['num_runs_is_found_best'] = num_runs_is_found_best
        return_dict['num_samples_is_found_best'] = num_samples_is_found_best
        return_dict['num_samples_is_found_best_per_run'] = num_samples_is_found_best_per_run
        return return_dict

    success_dict = {}
    for id in list(dict_for_df.keys())[:]:
        #print(id)
        samplesets = dict_for_df[id]['custom']['sampleset']
        eval_dict = _evaluate_success_rate(exact_sols=exact_sols, sample_sols=samplesets, n_exact=n_exact_sols_to_compare, n_samples=n_samples_to_compare, is_print_sols=False, is_print_meta=False)
        #print(eval_dict)
        success_dict[id] = {'is_found_best': np.any([val['is_found_best'] for val in eval_dict['submissions'].values()]), **eval_dict}
    return success_dict

def return_bar_plot(success_dict:dict=None, n_samples_to_compare:int=0, n_exact_sols_to_compare:int=0):
    if success_dict==None:
        raise ValueError('success_dict is required, it may be obtained by .extract_success_dict')
    elif n_samples_to_compare==0 or n_exact_sols_to_compare==0:
        raise ValueError('all inputs are required')
    def _x_ticks_grouped(bar_values, gsize, group_dist=0.3):
        #group_dist = 0.3  # the distance between groups
        multiplier = 0
        rects = []
        for chunk in np.array_split(np.array(bar_values),gsize):
            #x = np.linspace(0,1-group_dist,len(chunk), endpoint=False)
            #offset = 1 * multiplier
            group_dist = 3
            x = np.linspace(0,len(bar_values)/gsize-group_dist,len(chunk), endpoint=False)
            offset = gsize * multiplier
            rects.extend(x + offset)
            multiplier += 1
        print(bar_values)
        print(rects)
        return rects
    
    x_axis_labels = [key for key in success_dict.keys()]
    bar_values_param_found_best = [val['is_found_best'] for val in success_dict.values()]
    bar_values_run_found_best = [val2 for val in success_dict.values() for val2 in val['is_found_best_per_run']]
    bar_values_submission_found_best = [val2['is_found_best'] for val in success_dict.values() for val2 in val['submissions'].values()]
    bar_values_param_num_matched = [val['num_matched'] for val in success_dict.values()]
    bar_values_run_num_matched = [val2 for val in success_dict.values() for val2 in val['num_matched_per_run']]
    bar_values_submission_num_matched = [val3 for val in success_dict.values() for val2 in val['num_matched_per_sub_per_run'] for val3 in val2]
    bar_values_param_samples = [val['num_samples'] for val in success_dict.values()]
    bar_values_run_samples = [val2 for val in success_dict.values() for val2 in val['num_samples_per_run']]
    bar_values_submission_samples= [val3 for val in success_dict.values() for val2 in val['num_samples_per_sub_per_run'] for val3 in val2]
    bar_values_num_runs_found_best = [val['num_runs_is_found_best'] for val in success_dict.values()]
    bar_values_num_samples_found_best = [val['num_samples_is_found_best'] for val in success_dict.values()]
    bar_values_num_samples_found_best_per_run = [val2 for val in success_dict.values() for val2 in val['num_samples_is_found_best_per_run']]
    fig, axs = plt.subplots(nrows=5, ncols=3)
    axs[0,0].set_title('did paramset find \n best solution')
    axs[0,0].bar(range(len(bar_values_param_found_best)), bar_values_param_found_best)
    axs[0,1].set_title('did run find \n best solution')
    axs[0,1].bar(range(len(bar_values_run_found_best)), bar_values_run_found_best)
    #axs[0,1].bar(x_ticks_grouped(bar_values_run_found_best, len(success_dict)), bar_values_run_found_best, width=10/(len(bar_values_run_found_best)))
    #axs[0,0].set_xticks(axs[0,0].get_xticks())
    #_ = axs[0,0].set_xticklabels(axs[0,0].get_xticklabels(), rotation=45)
    axs[0,2].set_title('did submission of paramset \n find best solution')
    axs[0,2].bar(range(len(bar_values_submission_found_best)), bar_values_submission_found_best)
    axs[1,0].set_title('overlap of best {n_s} samples and \n best {n_e} correct solutions per paramset'.format(n_s=n_samples_to_compare, n_e=n_exact_sols_to_compare))
    axs[1,0].bar(range(len(bar_values_param_num_matched)), bar_values_param_num_matched)
    axs[1,1].set_title('overlap of best {n_s} samples and \n best {n_e} correct solutions per run'.format(n_s=n_samples_to_compare, n_e=n_exact_sols_to_compare))
    axs[1,1].bar(range(len(bar_values_run_num_matched)), bar_values_run_num_matched)
    axs[1,2].set_title('overlap of best {n_s} samples and \n best {n_e} correct solutions per submission'.format(n_s=n_samples_to_compare, n_e=n_exact_sols_to_compare))
    axs[1,2].bar(range(len(bar_values_submission_num_matched)), bar_values_submission_num_matched)
    axs[2,0].set_title('num samples per paramset')
    axs[2,0].bar(range(len(bar_values_param_samples)), bar_values_param_samples)
    axs[2,1].set_title('num samples per run')
    axs[2,1].bar(range(len(bar_values_run_samples)), bar_values_run_samples)
    axs[2,2].set_title('num samples per submission')
    axs[2,2].bar(range(len(bar_values_submission_samples)), bar_values_submission_samples)
    axs[3,0].set_title('num runs find best solution')
    axs[3,0].bar(range(len(bar_values_num_runs_found_best)), bar_values_num_runs_found_best)
    axs[3,1].set_title('num samples find best solution')
    axs[3,1].bar(range(len(bar_values_num_samples_found_best)), bar_values_num_samples_found_best)
    axs[3,2].set_title('num samples per run find best solution')
    axs[3,2].bar(range(len(bar_values_num_samples_found_best_per_run)), bar_values_num_samples_found_best_per_run)

    fig.tight_layout()

    return fig

def extract_started_sets_from_success_dict(success_dict:dict=None, dict_info_read:dict=None):
    if success_dict==None:
        raise ValueError('success_dict is required')
    elif dict_info_read==None:
        raise ValueError('dict_info_read is required for comparison, this might be omitted in the future')
    
    started_sets = [s.encode('utf-8') for s in success_dict.keys()]
    ids_study = np.zeros(len(started_sets),dtype=int)
    for i, id_started in enumerate(started_sets):
        ids_study[i] = np.argwhere(id_started == dict_info_read['study']['data']['identifiers'])[0,0]
    study_matched_started_ids = np.lib.recfunctions.rec_append_fields(dict_info_read['study']['data'][ids_study], 'started sets', started_sets)
    
    return started_sets, study_matched_started_ids

def generate_sampler_array_for_plots(success_dict:dict=None, results_names:list=[], started_sets:list=[], study_matched_started_ids:np.recarray=None, n_samples_to_compare:int=0, n_exact_sols_to_compare:int=0):
    if success_dict==None:
        raise ValueError('success_dict is required')
    elif results_names==[] or started_sets==[] or not isinstance(study_matched_started_ids,np.recarray) or n_samples_to_compare==0 or n_exact_sols_to_compare==0:
        raise ValueError('all inputs are required')
    
    name_fraction_matched = f'fraction_samples_matched_{n_samples_to_compare}_samps_{n_exact_sols_to_compare}_sols'
    results_names.append('fraction_samples_is_found_best')
    results_names.append(name_fraction_matched)
    print(results_names)

    sampler_array = np.zeros((len(started_sets),len(results_names)))
    for key, value in success_dict.items():
        id_row = np.argwhere(key.encode('utf-8') == study_matched_started_ids['identifiers'])[0,0]
        for key2, value2 in value.items():
            if key2 in results_names:
                id_col = np.argwhere(key2 == np.array(results_names))
                sampler_array[id_row,id_col] = value2
    id_tmp_val = np.argwhere('fraction_samples_is_found_best' == np.array(results_names))[0,0]
    id_tmp_num = np.argwhere('num_samples_is_found_best' == np.array(results_names))[0,0]
    id_tmp_den = np.argwhere('num_samples' == np.array(results_names))[0,0]
    sampler_array[:,id_tmp_val] = sampler_array[:,id_tmp_num] / sampler_array[:,id_tmp_den]
    id_tmp_val = np.argwhere(name_fraction_matched == np.array(results_names))[0,0]
    id_tmp_num = np.argwhere('num_samples_matched' == np.array(results_names))[0,0]
    sampler_array[:,id_tmp_val] = sampler_array[:,id_tmp_num] / sampler_array[:,id_tmp_den]

    return sampler_array

def return_plots(study_name:str='', study_matched_started_ids:np.recarray=None\
        , gp_mean:np.ndarray=None, results_names:list=[], dict_info_read:dict=None\
        , sampler_array:np.ndarray=None, colour_label:str='', dir_name_path_plots:str=''):
    if study_name == 'sub_2':
        plot_array = study_matched_started_ids['sets'].view(np.float64)
    else:
        plot_array = study_matched_started_ids['sets'].view((np.float64\
                    , len(study_matched_started_ids['sets'].dtype.names))).copy()

    cmap = plt.colormaps['viridis']

    nrows_uni_log = plot_array.shape[1]
    ncols_uni_log = gp_mean.shape[1]
    nrows_uni_lin = plot_array.shape[1]
    ncols_uni_lin = gp_mean.shape[1]
    nrows_bi_log = plot_array.shape[1]
    ncols_bi_log = plot_array.shape[1]
    ncols_bi_lin = plot_array.shape[1]

    fig2, axs2 = plt.subplots(nrows=nrows_uni_log+nrows_uni_lin+nrows_bi_log, ncols=ncols_uni_log \
                              , figsize=(2.5*ncols_uni_log, 2.5*(nrows_uni_log+nrows_uni_lin+nrows_bi_log))\
                              , layout='compressed')

    print('Figure dimension:', axs2.shape)
    for row_id in range(nrows_uni_log):
        for col_id in range(ncols_uni_log):
            if study_name == 'sub_2': axs2[row_id,col_id].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,row_id], gp_mean[:,col_id])
            #else : axs2[row_id,col_id].scatter(dict_info_read['study']['data']['sets'].view(np.float64, len(dict_info_read['study']['data']['sets'].dtype.names))[:,row_id], gp_mean[:,col_id])
            else : axs2[row_id,col_id].scatter(dict_info_read['study']['data']['sets'].copy().view((np.float64, len(dict_info_read['study']['data']['sets'].dtype.names)))[:,row_id], gp_mean[:,col_id])
            axs2[row_id,col_id].set_title(f'{results_names[col_id]} vs.\
    \n{dict_info_read['study']['data']['sets'].dtype.names[row_id]}')
            axs2[row_id,col_id].set_xscale('log')
            axs2[row_id,col_id].set_yscale('log')

            if study_name == 'sub_2': axs2[row_id+nrows_uni_log,col_id].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,row_id], gp_mean[:,col_id])
            #else: axs2[row_id+nrows_uni_log,col_id].scatter(dict_info_read['study']['data']['sets'].view(np.float64, len(dict_info_read['study']['data']['sets'].dtype.names))[:,row_id], gp_mean[:,col_id])
            else: axs2[row_id+nrows_uni_log,col_id].scatter(dict_info_read['study']['data']['sets'].copy().view((np.float64, len(dict_info_read['study']['data']['sets'].dtype.names)))[:,row_id], gp_mean[:,col_id])
            axs2[row_id+nrows_uni_log,col_id].set_title(f'{results_names[col_id]} vs.\
    \n{dict_info_read['study']['data']['sets'].dtype.names[row_id]}')

    
    colour_id = np.argwhere(colour_label == np.array(results_names))[0,0]
    print('colour_id =', colour_id)

    for row_id in range(nrows_bi_log):
        for col_id in range(ncols_bi_log):
            #print(row_id, col_id, ncols_bi_log)
            axs2[row_id+nrows_uni_log+nrows_uni_lin,col_id].scatter(plot_array[:,row_id], plot_array[:,col_id], \
                                        c=sampler_array[:,colour_id], cmap=cmap)
            axs2[row_id+nrows_uni_log+nrows_uni_lin,col_id].set_title(f'{dict_info_read['study']['data']['sets'].dtype.names[col_id]} vs.\
    \n{dict_info_read['study']['data']['sets'].dtype.names[row_id]}\
    ')#\n(c={results_names[colour_id]})')
            axs2[row_id+nrows_uni_log+nrows_uni_lin,col_id].set_xscale('log')
            axs2[row_id+nrows_uni_log+nrows_uni_lin,col_id].set_yscale('log')
            # linear axis scale
            axs2[row_id+nrows_uni_log+nrows_uni_lin,col_id+ncols_bi_lin].scatter(plot_array[:,row_id], plot_array[:,col_id], \
                                        c=sampler_array[:,colour_id], cmap=cmap)
            axs2[row_id+nrows_uni_log+nrows_uni_lin,col_id+ncols_bi_lin].set_title(f'{dict_info_read['study']['data']['sets'].dtype.names[col_id]} vs.\
    \n{dict_info_read['study']['data']['sets'].dtype.names[row_id]}\
    ')#\n(c={results_names[colour_id]})')
        axs2[row_id+nrows_uni_log+nrows_uni_lin,-1].axis('off')





    # Hist is problematic with fractions, as we would need to compute fractions of samples/etc w.r.t. the number of occurences in the respective bin, which can not be done by matplotlib directly
    #for row_id in range(nrows_uni_hist):
    #    for col_id in range(ncols_uni_hist):
    #        axs2[row_id+nrows_uni, col_id].hist(x = study_matched_started_ids['sets'].view(np.float64)[:,row_id],
    #                                  weights = sampler_array[:,col_id],\
    #                                  bins = 20)
    #        axs2[row_id+nrows_uni,col_id].set_title(f'{results_names[col_id]} vs.\
    #\n{dict_info_read['study']['data']['sets'].dtype.names[row_id]}')


    #fig2.tight_layout() # does not work with colorbar, so instead introduced "layout='compressed'" in figure creation
    sm = cm.ScalarMappable(norm=plt.Normalize(sampler_array[:,colour_id].min(),sampler_array[:,colour_id].max()),cmap=cmap)
    sm.set_array([])
    fig2.colorbar(sm, ax=axs2[nrows_uni_log+nrows_uni_lin:,-1].ravel().tolist(), label=colour_label)
    fig2.savefig(dir_name_path_plots + f'fig_{study_name}_{results_names[colour_id]}.svg')
    #fig2.savefig(f'./01_out/fig_colorscale_legend.svg')

    #axs2[0,0].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,0], mean[:,1])
    #axs2[0,1].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,1], mean[:,1])
    #axs2[0,2].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,2], mean[:,1])
    #axs2[0,3].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,3], mean[:,1])
    ##axs2[0,0].set_xscale('log')
    ##axs2[0,1].set_xscale('log')
    ##axs2[0,2].set_xscale('log')
    ##axs2[0,3].set_xscale('log')
    #axs2[0,0].set_ylim((0., 10.))
    #axs2[0,1].set_ylim((0., 10.))
    #axs2[0,2].set_ylim((0., 10.))
    #axs2[0,3].set_ylim((0., 10.))
    #axs2[1,0].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,0], mean[:,2])
    #axs2[1,1].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,1], mean[:,2])
    #axs2[1,2].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,2], mean[:,2])
    #axs2[1,3].scatter(dict_info_read['study']['data']['sets'].view(np.float64)[:,3], mean[:,2])
    ##axs2[1,0].set_xscale('log')
    ##axs2[1,1].set_xscale('log')
    ##axs2[1,2].set_xscale('log')
    ##axs2[1,3].set_xscale('log')
    ##axs2[1,0].set_ylim((0., 10.))
    ##axs2[1,1].set_ylim((0., 10.))
    ##axs2[1,2].set_ylim((0., 10.))
    #axs2[1,3].set_ylim((0., 10.))
    
    
    
    
    
    if study_name == 'sub_2': dict_parallel_plot_inputs = {name: study_matched_started_ids['sets'][name].flatten()for name in study_matched_started_ids['sets'].dtype.names}
    else : dict_parallel_plot_inputs = {name: study_matched_started_ids['sets'][name] for name in study_matched_started_ids['sets'].dtype.names}
    dict_parallel_plot_outputs = {results_names[i]: sampler_array[:,i] for i in range(len(results_names))}
    dict_parallel_plot = dict_parallel_plot_inputs | dict_parallel_plot_outputs # merges two dicts
    
    fig_pp1 = px.parallel_coordinates(
    dict_parallel_plot,
    color=results_names[colour_id],
    color_continuous_scale=px.colors.sequential.Viridis)
    fig_pp1.write_html(dir_name_path_plots + f'fig_{study_name}_{results_names[colour_id]}_parallel_plot.html')
    
    
    
    
    return fig2, fig_pp1



