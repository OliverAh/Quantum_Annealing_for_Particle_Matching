import h5py
import os.path
import numpy as np
import concurrent.futures.thread
#import multiprocessing # multiprocessing is a hell to fiddle with in Jupyter noteboks inside VS Code, so no imlementation for now 
import pickle
import codecs

def _is_set_identifier_in_file_already(file_handle, set_identifier):
    if set_identifier in file_handle.keys():
        return True
    else:
        return False


def write_to_hdf5_file(file_name_path='', dict_data: dict = {}, data_name: str = '', name_suffix: str = '', overwrite_data_in_file=False, track_order=False) -> None:
    """ Writes data to a hdf5 file. If the file does not exist, it will be created. 
    If the file exists, the data will be written to it. 
    If the data already exists in the file, it might or might not be overwritten, depending on variable (default is not to overwrite). 
    params:
        - file_name_path: str, default ''. Path to the file. If it contains only the file name, it will be saved in the current working directory.
        - dict_data: dict, default {}. Data to be written to the file.
        - data_name: str, default ''. Name of the data to be written to the file. Must be one of ['particles', 'sampler', 'embedding', 'composite', 'sampleset', 'sampleset_sa'].
        - overwrite_data_in_file: bool, default False. If True, the data will be overwritten if the file already exists and it already contains data of the provided data_name.
            If false, the data will not be overwritten and an error will be raised if the file already exists and it already contains data of the provided data_name."""
    list_valid_data_names = ['particles', 'sampler', 'embedding', 'composite', 'sampleset', 'sampleset_sa', 'custom']
    file_mode = '' # will be adjusted while checking file and settings
    data_name_int = data_name + name_suffix
    # check for valid file_name_path
    if file_name_path[-3:] != '.h5':
        raise ValueError('file_name_path must end with .h5, i.e. must be a hdf5 file')
    # check for valid data_name
    if data_name not in list_valid_data_names:
        raise ValueError('data_name must be one of {}'.format(list_valid_data_names))
    # check for valid data type (top most level only)
    if not isinstance(dict_data, dict):
        raise TypeError('dict_data must be of type dict')
    # check for empty data dict
    if len(dict_data) == 0: # Return the number of items in the dictionary
        raise ValueError('dict_data is empty, might empty dict was provided or no dict was provided')
    # check settings for overwriting file and/or data
    if os.path.isfile(file_name_path):
        with h5py.File(file_name_path, 'r') as f:
            if data_name_int in f.keys():
                if data_name == 'sampleset' or data_name == 'sampleset_sa' or 'composite':
                    if data_name_int in f.keys() and _is_set_identifier_in_file_already(f[data_name], dict_data['set_identifier']):
                        if overwrite_data_in_file==False:
                            print('Sampleset with set identifier {} already exists in file. Set overwrite_data_in_file=True to overwrite data in file. Sorting out...'.format(dict_data['set_identifier']))
                            while _is_set_identifier_in_file_already(f[data_name], dict_data['set_identifier']):
                                _id = dict_data['set_identifier'].decode('utf-8')
                                print(_id)
                                _id += '_z'
                                print(_id)
                                dict_data['set_identifier'] = _id.encode('utf-8')
                                print(dict_data['set_identifier'])
                            print('New set identifier: {}'.format(dict_data['set_identifier']))
                            file_mode = 'r+'
                        elif overwrite_data_in_file==True:
                            file_mode = 'r+'
                    else:
                        file_mode = 'r+'
                else:
                    if overwrite_data_in_file==False:
                        raise ValueError('Data already exists in file. Set overwrite_data_in_file=True to overwrite data in file.')
                    elif overwrite_data_in_file==True:
                        file_mode = 'r+' # Read/write, file must exist
                    else:
                        raise ValueError('overwrite_data_in_file must be of type bool')
            else:
                if overwrite_data_in_file==True:
                    print('File does not yet contain data with this name but overwrite_data_in_file=True is set. I will continue with writing the data to it, so this warning might be igrnored')
                    file_mode = 'r+' # Read/write, file must exist
                elif overwrite_data_in_file==False:
                    file_mode = 'r+' # Read/write, file must exist
                else:
                    raise ValueError('overwrite_data_in_file must be of type bool')
                
    else:
        os.makedirs(os.path.dirname(file_name_path), exist_ok=True) # creates only dir, not file 
        if overwrite_data_in_file==True:
            print('File does not exist, but overwrite_data_in_file=True was set. I will continue with generating a new file and writing the data to it, so this warning might be ignored')
            file_mode = 'w-' #Create file, fail if exists
        elif overwrite_data_in_file==False:
            file_mode = 'w-' #Create file, fail if exists
        else:
            raise ValueError('overwrite_data_in_file must be of type bool')

    try:
        with h5py.File(file_name_path, mode = file_mode, track_order=track_order) as file_handle:
            if overwrite_data_in_file==False:
                file_handle.require_group(data_name_int) # Would not overwrite if exists already (raise TypeError) 
                pass
            elif overwrite_data_in_file==True:
                if data_name_int in file_handle.keys():
                    del file_handle[data_name_int]
                file_handle.create_group(data_name_int) # Should overwrite if exists already, doesnt for some reason
            else:
                raise ValueError('overwrite_data_in_file must be of type bool \n This should have been catched earlier, so this is a bug')

            
            def write_dict_to_hdf5_recursively(file_handle_recur, dict_data_recur: dict = {}, data_name_recur: str = ''):
                for key, value in dict_data_recur.items():
                    #print(key, type(value), value)
                    if isinstance(value, dict):
                        file_handle_recur[data_name_recur].create_group(key)
                        write_dict_to_hdf5_recursively(file_handle_recur, dict_data_recur=value, data_name_recur=data_name_recur+'/'+key)
                    else:
                        key_2 = key
                        if not isinstance(key, (str, bytes)):
                            key_2 = str(key)
                        
                        if value is None:
                            file_handle_recur[data_name_recur].create_dataset(key_2, data='None')
                        elif isinstance(value, (int, float, bool)):
                            file_handle_recur[data_name_recur].create_dataset(key_2, data=value)
                        elif isinstance(value, set):
                            file_handle_recur[data_name_recur].create_dataset(key_2, data=list(value))
                        elif isinstance(value, (np.ndarray, str, list, tuple)):
                            
                            try:
                                file_handle_recur[data_name_recur].create_dataset(key_2, data=value)
                            except:
                                print('I Error in writing data to file. Trying to convert data to string and write it to file.')
                                print(key_2, type(key_2))
                                print(value, type(value))
                                try:
                                    file_handle_recur[data_name_recur].create_dataset(key_2, data=str(value))
                                except:
                                    try:
                                        _tmp_data = pickle.dumps(value)
                                        _tmp_data = _tmp_data.encode('utf-8')
                                        file_handle_recur[data_name_recur].create_dataset(key_2, data=_tmp_data)
                                    except:
                                        print('Could not write data to file. Skipping...', key_2, type(value), value)
                        #elif not isinstance(key, str):
                        #    file_handle_recur[data_name_recur].create_dataset(str(key), data=value)
                        elif isinstance(value, concurrent.futures.thread.ThreadPoolExecutor):
                            print('elif concurrent.futures.thread.ThreadPoolExecutor')
                            print(key_2, type(value), value)
                            continue
                        else:
                            print(key_2, type(value), value)
                            try:
                                file_handle_recur[data_name_recur].create_dataset(key_2, data=value)
                            except:
                                print('II Error in writing data to file. Trying to convert data to string and write it to file.')
                                print(key_2, type(key_2))
                                print(value, type(value))
                                try:
                                    file_handle_recur[data_name_recur].create_dataset(key_2, data=str(value))
                                except:
                                    try:
                                        _tmp_data = pickle.dumps(value)
                                        _tmp_data = _tmp_data.encode('utf-8')
                                        file_handle_recur[data_name_recur].create_dataset(key_2, data=_tmp_data)
                                    except:
                                        print('Could not write data to file. Skipping...', key_2, type(value), value)
            
            if data_name == 'embedding':
                dict_data_recur = {str(key): value for key, value in dict_data.items()} # keys are just integers, so convert to string
            elif data_name == 'sampleset' or data_name == 'sampleset_sa': # make changes to what is returned by sampleset.to_serializable here because I like the style of the serialized version and want to keep it as much of it as possible 
                dict_data_recur = dict_data[data_name]
                data_name_int += '/' + dict_data['set_identifier'].decode('utf-8')
                file_handle.require_group(data_name_int)
                if 'warnings' in dict_data_recur['info'].keys():
                    for i in range(len(dict_data_recur['info']['warnings'])):
                        tmp_type = dict_data_recur['info']['warnings'][i]['type']
                        dict_data_recur['info']['warnings'][i]['type'] = str(tmp_type.__name__)
                    dict_data_recur['info']['warnings'] = {'warning_{}'.format(i): warning for i, warning in enumerate(dict_data_recur['info']['warnings'])}
            else:
                dict_data_recur = dict_data
            write_dict_to_hdf5_recursively(file_handle, dict_data_recur=dict_data_recur, data_name_recur='/'+data_name_int)
    except:
        raise

def read_embedding_from_hdf5_file(file_name_path='', data_name: str = 'embedding') -> dict :
    
    def _to_dict(obj):
        #print(obj)
        #print(len(obj.keys()))
        #print(obj.items())
        return {int(key): list(value[()]) for key, value in obj.items()}
    
    try:
        with h5py.File(file_name_path, 'r') as file_handle:
            #embedding = {key: value[()] for key, value in embedding.items()}
            #print('Filename: ', file_handle.filename)
            #for key in file_handle.keys():
            #    print(' -', key, ' ', file_handle[key])
            #print()
            if data_name in file_handle.keys():
                return _to_dict(file_handle[data_name])
            else:
                raise ValueError('File does not contain data with name {}'.format(data_name))

    except:
        raise ValueError('Could not read embedding from file {}'.format(file_name_path))
    
def _read_groups_recursively_from_hdf5_object(obj: h5py.Group, name: str='') -> dict :
    """ Reads groups and their sub-structures recursively from a hdf5 file."""

    intermediate_dict = {}
    if obj.attrs.keys():
        intermediate_dict.update({'attrs': {key: value for key, value in obj.attrs.items()}})
    for key, value in obj.items():
        if isinstance(value, h5py.Group):
            intermediate_dict.update(_read_groups_recursively_from_hdf5_object(value, name=key))
        elif isinstance(value, h5py.Dataset):
            if value.attrs.keys():
                intermediate_dict.update({key: {'attrs': {attr_key: attr_value for attr_key, attr_value in value.attrs.items()}, 'data': value[()]}})
            else:
                intermediate_dict.update({key: {'attrs': {}, 'data': value[()]}})
    
    if name == '':
        return intermediate_dict
    else:
        return {name: intermediate_dict}

def read_info_from_hdf5_file(file_name_path='', infoset_name: str = '', driver: str ='') -> dict :
    """ Reads data for variation study from a hdf5 file. To a dict of dicts where each group forms a dict. 
    Was initially planned for info_file only, but can also be used for sample-data etc. 
    driver='core' reads the whole file into memery and then processes it from there, which is 10%-50% faster, but one needs to keep an eye on memory.
    See > h5py >> File Objects >> File drivers < for more information."""
    if not isinstance( file_name_path, str):
        if str(file_name_path)[-3:] != '.h5':
            raise ValueError('file_name_path must end with .h5, i.e. must be a hdf5 file')
    elif file_name_path[-3:] != '.h5':
        raise ValueError('file_name_path must end with .h5, i.e. must be a hdf5 file')
    try:
        if driver == '':
            driver = None
        elif isinstance(driver, str) or driver == None:
            pass
        else:
            raise ValueError('driver must be of type str or None')

        with h5py.File(file_name_path, 'r', driver=driver) as file_handle:
            if infoset_name == '':
                return _read_groups_recursively_from_hdf5_object(file_handle)
            elif infoset_name in file_handle.keys():
                return _read_groups_recursively_from_hdf5_object(file_handle[infoset_name])
            else:
                raise ValueError('File does not contain data with name {}'.format(infoset_name))
    except:
        raise ValueError('Could not read info from file {}'.format(file_name_path))