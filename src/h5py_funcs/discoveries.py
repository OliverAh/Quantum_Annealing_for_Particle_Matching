import h5py
import dimod
import dwave

def discover_hdf5_file_structure(file_handle, print_dataset_values=False):
    """Goes through all groups and datasets in a HDF5 file recursively and prints their structure and attributes in an 
     easy-to-read format. 
     params:
        - file: h5py.File object
        - print_dataset_values: bool, default False. If True, prints additionally the values of all datasets in the file."""
    def print_dataset(obj, indent=''):
        print(obj[()])
    
    def print_attrs(obj, indent=''):
        attr_list = [key for key in obj.attrs.keys()]
        if attr_list == []:
            print(indent, '- No attributes')
        else:
            print(indent, '- Attributes:')
            for key, val in obj.attrs.items():
                print(indent, '  - ', key, ':', val)
        

    def discover_hdf5_obj(obj, indent=''):
        if isinstance(obj, h5py.Dataset):
            print(indent, '- Dataset:', obj.name)
            print_attrs(obj, indent=indent+'  ')
            print(indent, '  - shape: ', obj.shape)
            print(indent, '  - size:  ', obj.size)
            print(indent, '  - ndim:  ', obj.ndim)
            print(indent, '  - dtype: ', obj.dtype)
            print(indent, '  - nbytes:', obj.nbytes)
            print(indent, '   ', obj)
            if print_dataset_values:
                print_dataset(obj, indent=indent+'  ')
        elif isinstance(obj, h5py.Group):
            print(indent, '- Group:', obj.name)
            print_attrs(obj, indent=indent+' |')
            for key in obj.keys():
                discover_hdf5_obj(obj[key], indent=indent+' |')

    print('Filename: ', file_handle.filename)
    for key in file_handle.keys():
        print(' -', key, ' ', file_handle[key])
    print()
    for key in file_handle.keys():
        discover_hdf5_obj(file_handle[key])

def _stringify(val, key, indent):
    return str(val).replace('\n','\n'+indent+' '*(6+len(key)))

union_types_discover_hdf5 = dwave.system.samplers.dwave_sampler.DWaveSampler | dwave.system.composites.embedding.FixedEmbeddingComposite | dimod.sampleset.SampleSet | dict | list
def discover_obj_data_for_hdf5(obj: union_types_discover_hdf5, indent=''):
    """Goes through all levels of an object and prints their structure and attributes in an easy-to-read format.
       params:
          - obj: object to discover. Type must be one of [dwave.system.samplers.dwave_sampler.DWaveSampler, dwave.system.composites.embedding.FixedEmbeddingComposite, dimod.sampleset.SampleSet, dict or list].
            With obj being given as a dict from the customized __getstate__ function, one can check what information would be written to an hdf5 file.
          - indent: string, default ''. Used for recursive calls to indent the output. Does not need to be given by the user."""
    if not isinstance(obj, union_types_discover_hdf5):
        raise TypeError('obj is of type {}, but must be one of {}'.format(type(obj), union_types_discover_hdf5))
    
    def print_dict(obj_dict: dict, indent=''):
        if len(obj_dict) == 0: # Return the number of items in the dictionary
            print(indent, '- This dict is empty.')
        else:
            for key, value in obj_dict.items():
                if isinstance(value, dwave.cloud.client.qpu.Client):
                    print(indent, '-', key, type(value))
                    discover_qpu_Client(value, indent=indent+' |')
                elif isinstance(value, dwave.cloud.solver.StructuredSolver):
                    print(indent, '-', key, type(value))
                    discover_StructuredSolver(value, indent=indent+' |')
                elif isinstance(value, dict):
                    print(indent, '-', key, type(value))
                    print_dict(value, indent=indent+' |')
                elif isinstance(value, list) and key=='children':
                    discover_list(key, value, indent=indent)
                elif isinstance(value, dwave.cloud.config.models.ClientConfig):
                    print(indent, '-', key, type(value))
                    discover_ClientConfig(value, indent=indent+' |')
                elif isinstance(value, dwave.system.samplers.dwave_sampler.DWaveSampler):
                    discover_obj_data_for_hdf5(value, indent=indent+' -')
                else:    
                    print(indent, '-', key, ':', _stringify(value, key, indent), type(value))
        
    def discover_ClientConfig(obj, indent=''):
        print_dict(obj.__dict__, indent=indent)

    def discover_list(key, value, indent=''):
        if key in ['_nodelist', '_edgelist']:
            print(indent, '-', key, value)
        elif key == 'children' and all([isinstance(value[i], dict) for i in range(len(value))]):
            print(indent, '-', key, type(value))
            for i, child in enumerate(value):
                print(indent, '| -', 'sampler', type(value[i]))
                print_dict(child, indent=indent+' | |')
        elif key == 'children' and all([isinstance(value[i], dwave.system.samplers.dwave_sampler.DWaveSampler) for i in range(len(value))]):
            print(indent, '-', key, type(value))
            for i, child in enumerate(value):
                #print(indent, '| -', type(value[i]))
                discover_obj_data_for_hdf5(child, indent=indent+' | -')
        else:
            print(indent, '-', key, type(value))
            forward_obj = {'list_item{}'.format(i): item for i, item in enumerate(value)}
            print_dict(forward_obj, indent=indent+' |')

    def discover_qpu_Client(obj, indent=''):
        print_dict(obj.__dict__, indent=indent)

    def discover_StructuredSolver(obj, indent=''):
        print_dict(obj.__dict__, indent=indent)
    
    print(indent, type(obj))
    if indent != '' and indent[-1]=='-': indent = indent[:-2] # if evaluates to true if discover_obj_data_for_hdf5 calls itself from within its print_dict, then this prevents the ' -' to show up in every print following the previous line
    indent += ' |'
    if not isinstance(obj, dict):
        for key, value in obj.__dict__.items():
            if isinstance(value, dwave.cloud.client.qpu.Client):
                print(indent+' -', key, type(value))
                discover_qpu_Client(value, indent=indent+' |')
            elif isinstance(value, dwave.cloud.solver.StructuredSolver):
                print(indent+' -', key, type(value))
                discover_StructuredSolver(value, indent=indent+' |')
            elif isinstance(value, dict):
                print(indent+' -', key, type(value))
                print_dict(value, indent=indent+' |')
            elif isinstance(value, list):
                discover_list(key, value, indent=indent)
            elif isinstance(value, union_types_discover_hdf5) and not isinstance(value, dict | list):
                discover_obj_data_for_hdf5(value, indent=indent+' |')
            else:
                print(indent+' -', key, type(value))
    elif isinstance(obj, dict):
        # is only meant to compare data available, and data that would be returned by __getstate__ used for e.g. writing to hdf5 file
        print_dict(obj, indent=indent)
    else:
        raise TypeError('obj is of type {}, but must be one of {}'.format(type(obj), union_types_discover_hdf5))