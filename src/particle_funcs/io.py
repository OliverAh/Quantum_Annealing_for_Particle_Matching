import os
import pathlib
import numpy as np
import h5py
from src.h5py_funcs.io import _read_groups_recursively_from_hdf5_object as read_recurs_hdf5
import tqdm


def read_dem_data(folder_path:pathlib.Path|str=''):
    """
    Read DEM data from a folder and return the data as a dict.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the DEM data.

    Returns
    -------
    dict
        The DEM data as a numpy array.
    """
    # Read the DEM data from the folder
    file_list_trajectories = os.listdir(pathlib.Path(folder_path).joinpath('simulation'))
    file_list_trajectories = [file for file in file_list_trajectories if file.endswith('.rhs')]
    #print(file_list_trajectories)
    #print(len(file_list_trajectories))
    
    dem_data = {}
    with h5py.File(pathlib.Path(folder_path).joinpath('simulation').joinpath(file_list_trajectories[0]), 'r') as file_handle:
        dem_data.update({'num_particles': file_handle['Particles'].attrs['n_total_particles'][0]})
        dem_data.update({'particle_index': file_handle['Particles']['particles_index'][()]})
    for file in tqdm.tqdm(file_list_trajectories):
        with h5py.File(pathlib.Path(folder_path).joinpath('simulation').joinpath(file), 'r') as file_handle:
            dem_data.update({file: read_recurs_hdf5(file_handle)})
    return dem_data