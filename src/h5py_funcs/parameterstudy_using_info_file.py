""" Functions to conduct a parameterstudy using an info-file to deal with organizing different runs.
Workflow is as follows:
- first prepare info-file with 

 """




import h5py
import numpy as np
import os


def _explore_np_datetime64():
    a = np.datetime64('now')
    print(a)
    b = np.datetime_as_string(a)
    c = np.datetime_as_string(a, unit='s', timezone='local')
    print(b)
    print((c), type(c))
    print(np.datetime_as_string(np.datetime64('now'), unit='s', timezone='local'))
    d = np.datetime64('0001-01-01T00:00:00')
    e = np.datetime_as_string(d, unit='s', timezone='UTC')
    print(d)
    print(e)

def _recursicely_write_dict_to_hdf5_group(group, dict_to_write):
    for key, value in dict_to_write.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _recursicely_write_dict_to_hdf5_group(group=subgroup, dict_to_write=value)
        else:
            group.create_dataset(key, data=value)


def _current_datetime_as_string(offset=None):
    datetime = np.datetime64('now', 's')
    if offset is not None:
        datetime += np.timedelta64(int(offset), 'h')
    np_string = np.datetime_as_string(datetime, unit='s', timezone='local')
    np_string = np_string.encode('utf-8')
    return np_string

def _ensure_info_file(folder_path_name, info_file_name):
    if os.path.exists(os.path.join(folder_path_name, info_file_name)):
        print('info-file {} exists in folder {} already. This one will be used'.format(info_file_name, folder_path_name))
    else:
        try:
            with h5py.File(os.path.join(folder_path_name, info_file_name), 'w-') as f: #w- or x Create file, fail if exists
                print('Created new info-file {} in folder {}.'.format(info_file_name, folder_path_name))
        except:
            raise LookupError('info-file {} does not exist in folder {}, but something went wrong and it could not be created.'.format(info_file_name, folder_path_name))

def _is_set_identifier_in_info_file_already(folder_path_name, set_identifier, info_file_name):
    if os.path.exists(os.path.join(folder_path_name, info_file_name)):
        try: 
            with h5py.File(os.path.join(folder_path_name, info_file_name), 'r') as f:
                if set_identifier in f.keys():
                    return True
                else:
                    return False
        except:
            raise LookupError('File {} exists in folder {}, but something went wrong and identifier {} could not be checked.'.format(info_file_name, folder_path_name, set_identifier))
    else:
        return False
    
def _generate_unused_set_identifier(used_identifiers):
    import random
    def gen():
        return 'zz_{0:010}'.format(random.randint(1, int(1e10))) # prefix zz_ so that info-file is show at top of file explorer 
    set_identifier = gen()
    while set_identifier in used_identifiers:
        set_identifier = gen()
    return set_identifier
def update_timestamp_in_info_file(file_name_path, set_identifier, name):
    _id = -99
    if name == 'start':
        _id = 1
    elif name == 'finish':
        _id = 2
    else:
        raise ValueError('name must be one of [\'start\', \'finish\']')

    try:
        with h5py.File(file_name_path, 'r+') as f:
            if name == 'start' and False == f['parametersets']['time_history'][set_identifier].attrs['ready']:
                    raise ValueError('Set {} exists in file {}, but it is not ready yet.'.format(set_identifier, file_name_path))
            f['parametersets']['time_history'][set_identifier][_id] = np.array(_current_datetime_as_string())
            if name == 'finish':
                f['parametersets']['time_history'][set_identifier].attrs.modify('finished', True)
    except:
        raise LookupError('Set {} exists in file {}, but something went wrong and timestamp could not be updated.'.format(set_identifier, file_name_path))

def _write_info_to_info_file(metadata_dict, problem_dict, parametersets_array, folder_path_name, info_file_name):
    try:
        with h5py.File(os.path.join(folder_path_name, info_file_name), 'r+', track_order=True) as f:
            f.create_group('parametersets', track_order=True)
            num_parametersets = np.shape(parametersets_array)[0]
            identifiers = ['zz_0000000000'] * num_parametersets
            for i, _ in enumerate(identifiers):
                identifiers[i] = _generate_unused_set_identifier(used_identifiers=identifiers)
            identifiers = np.array([_id.encode('utf-8') for _id in identifiers])
            #identifiers = np.array(identifiers)
            #f_data_2 = [(tuple(samples_salib[i,j] for j in range(np.shape(samples_salib)[1])), ids[i]) for i in range(np.shape(samples_salib)[0])]
            #f_dtype = [('a', [('aa','f'),('ab','f')],(1,)),# [(key, 'f64') for key in problem_dict['names']
            #           ('b', 'U10')]
            params_field_types = [(key, parametersets_array[i].dtype) for i, key in enumerate(problem_dict['names'])]
            rec_array_dtype = [('sets', params_field_types, (1,)), ('identifiers', identifiers.dtype)]
            # each row consists of a parameterset for a single run + its unique identifier
            # the columns can be accesed in two ways: -1 array.sets -> 2d array(num_parametersets, num_parameters)
            #                                         -2 array.sets.{param_name} -> 1d array(num_parametersets) of specified parameter
            #                                         -3 array.identifiers -> 1d array(num_parametersets) of unique identifiers
            #                                         identifiers are not returned by -1 and -2
            rec_array_data = [(tuple(param_set), identifiers[i])
                               for i, param_set in enumerate(parametersets_array)]

            rec_array = np.rec.fromrecords(rec_array_data, dtype=rec_array_dtype)
            f['parametersets'].create_dataset(name='study', data = rec_array, track_order=True)
            for key, value in metadata_dict.items():
                f['parametersets'].attrs.create(name=key, data=value) # name (String), data – Value of the attribute; will be put through numpy.array(data)., shape=None, dtype=None, shape and dty would overwrite values obtained from data
            for key, value in problem_dict.items():
                f['parametersets'].attrs.create(name=key, data=value)
            f['parametersets'].create_group(name='time_history', track_order=True)
            for _id in identifiers:
                f['parametersets']['time_history'].create_dataset(name=_id, shape=((3,)), dtype=np.array(_current_datetime_as_string()).dtype, track_order=True)
                f['parametersets']['time_history'].attrs.create(name='order', data=['creation', 'start sampling', 'finish sampling'])
                f['parametersets']['time_history'][_id][0] = np.array(_current_datetime_as_string())
                f['parametersets']['time_history'][_id][1] = np.array(_current_datetime_as_string(-1e6))
                f['parametersets']['time_history'][_id][2] = np.array(_current_datetime_as_string(-1e6))
                f['parametersets']['time_history'][_id].attrs.create(name='ready', data=True)
                f['parametersets']['time_history'][_id].attrs.create(name='finished', data=False)
            
                        
    except:
        raise LookupError('Info-file {} exists in folder {}, but something went wrong and data could not be written to it.'.format(info_file_name, folder_path_name))


def prepare_info_file(metadata_dict: dict={}, problem_dict: dict={}, parametersets_array: np.ndarray=np.array([]), folder_path_name: str='', info_file_name: str='parameterstudy_info.h5'):
    if len(metadata_dict) == 0:
        raise ValueError('metadata_dict is empty')
    for key, value in metadata_dict.items():
        if isinstance(value, dict):
            raise ValueError('metadata_dict must not contain dicts, but key {} is a dict'.format(key))
    for key, value in problem_dict.items():
        if isinstance(value, dict):
            raise ValueError('problem_dict must not contain dicts, but key {} is a dict'.format(key))
    if len(problem_dict) == 0:
        raise ValueError('problem_dict is empty')
    if np.shape(parametersets_array) == (0,):
        raise ValueError('parametersets_array is empty')
    if folder_path_name != '' and not os.path.exists(folder_path_name):
        print('Folder {} does not exist. Create it.'.format(folder_path_name))
        os.makedirs(folder_path_name)
    elif folder_path_name != '' and os.path.exists(folder_path_name):
        if os.path.exists(os.path.join(folder_path_name, info_file_name)):
            raise LookupError('In folder {} info-file {} already exists. This function does not overwrite data, so sort it out manually.'.format(folder_path_name, info_file_name))
    elif folder_path_name == '':
        raise ValueError('folder_path_name is empty but must be specified')
    _ensure_info_file(folder_path_name=folder_path_name, info_file_name=info_file_name)
    
    _write_info_to_info_file(metadata_dict=metadata_dict, problem_dict=problem_dict, parametersets_array=parametersets_array, folder_path_name=folder_path_name, info_file_name=info_file_name)



def verify_time_stamps_info_and_data_file(info_file_name_path: str='', data_file_name_path: str=''):
    import h5py
    import os

    if info_file_name_path == '':
        info_file_name_path = os.path.join('test_params_annealer', 'merged', 'parameterstudy_info_5__em_wo_migration_layout_09.h5')
    if data_file_name_path == '':
        data_file_name_path = os.path.join('test_params_annealer', 'merged', 'parameterstudy_data_5__em_wo_migration_layout_09.h5')

    _return_val = True
    try:
        with h5py.File(info_file_name_path, 'r') as f_info, h5py.File(data_file_name_path, 'r') as f_data:
            for key, value in f_info['parametersets']['time_history'].items():
                if value.attrs['finished'] == 1:
                    #print('finishd', key, value[...])
                    if key not in f_data['sampleset'].keys():
                        print('!!!!!!    not found in sampleset')
                        _return_val = False
                else:
                    if key in f_data['sampleset'].keys():
                        print('!!!!!!    not finished but found in sampleset')
                        _return_val = False
    except Exception as e:
        print(e)
        _return_val = False
    
    return _return_val