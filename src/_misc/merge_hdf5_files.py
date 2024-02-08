import h5py
import os
folder_path = os.path.join(os.path.relpath('test_params_annealer'), 'test_copy')
d_name = 'data_merged.h5'
s_1_name = 'parameterstudy_data_5__em_wo_migration_layout_09.h5'
s_2_name = 'parameterstudy_data_5__em_wo_migration_layout_09_2.h5'
s_3_name = 'parameterstudy_data_5__em_wo_migration_layout_09_3.h5'
d_name_path = os.path.join(folder_path, d_name)
s_1_name_path = os.path.join(folder_path, s_1_name)
s_2_name_path = os.path.join(folder_path, s_2_name)
s_3_name_path = os.path.join(folder_path, s_3_name)
print(d_name_path)
try:
    
    with h5py.File(d_name_path, 'a') as f_d, h5py.File(s_1_name_path, 'r') as f_s1, h5py.File(s_2_name_path, 'r') as f_s2, h5py.File(s_3_name_path, 'r') as f_s3:
    #with h5py.File(d_name_path, 'a') as f:
        print('here')
        f_d.create_group('sampleset_00')
        f_d.create_group('sampleset_01')
        f_d.create_group('sampleset_02')
        f_d.copy(f_s1['sampleset'], f_d['sampleset_00'])
        f_d.copy(f_s2['sampleset'], f_d['sampleset_01'])
        f_d.copy(f_s3['sampleset'], f_d['sampleset_02'])
except Exception as e:
    print('  WARNING: could not copy files')
    print(e)
    pass




import h5py
import os
folder_path = os.path.join(os.path.relpath('test_params_annealer'), 'merged')
d_name_data = 'parameterstudy_data_5__em_wo_migration_layout_09.h5'
s_1_name_data = 'zz_first_merge_parameterstudy_data_5__em_wo_migration_layout_09.h5'
d_name_info = 'parameterstudy_info_5__em_wo_migration_layout_09.h5'
s_1_name_info = 'zz_first_merge_parameterstudy_info_5__em_wo_migration_layout_09.h5'

d_name_path_data = os.path.join(folder_path, d_name_data)
s_1_name_path_data = os.path.join(folder_path, s_1_name_data)
d_name_path_info = os.path.join(folder_path, d_name_info)
s_1_name_path_info = os.path.join(folder_path, s_1_name_info)

print(d_name_path_data)
print(d_name_path_info)
try:
    
    with h5py.File(d_name_path_data, 'a') as f_d_data, h5py.File(s_1_name_path_data, 'r') as f_s_data, h5py.File(d_name_path_info, 'a') as f_d_info, h5py.File(s_1_name_path_info, 'r') as f_s_info:
        print('here')

        #f_d_info.create_group('parametersets')
        f_d_info['/'].copy(f_s_info['parametersets_00/parametersets'], f_d_info['/'])
        f_d_data['/'].copy(f_s_data['sampleset_00/sampleset'], f_d_data['/'])

        for sub_id in ['01', '02']:
        
            p_sets_info = f_s_info[f'parametersets_{sub_id}/parametersets']['study'][()]
            p_sets_info_merged = f_d_info['parametersets/study'][()]
            for _id in f_s_data[f'sampleset_{sub_id}/sampleset'].keys():
                for i in range(p_sets_info.size):
                    if _id == p_sets_info['identifiers'][i].decode('utf-8'):
                        #just for verification that 'study' arrays are euqal
                        for ii in range(p_sets_info_merged.size):
                            if p_sets_info_merged['sets'][ii] == p_sets_info['sets'][i]:
                                print(_id, i, ii, p_sets_info[i], p_sets_info_merged[ii])

                        id_merged = p_sets_info_merged['identifiers'][i].decode('utf-8')
                        attrs_merged = {key: value for key, value in f_d_info['parametersets/time_history'][id_merged].attrs.items()}
                        if attrs_merged['finished'] == True:
                            raise ValueError('  WARNING: attrs_merged[finished] == True')
                        info_info_merged = f_d_info['parametersets/time_history'][id_merged][()]
                        print('  ', id_merged, attrs_merged, info_info_merged)

                        f_d_data['sampleset'].copy(f_s_data[f'sampleset_{sub_id}/sampleset'][_id], f_d_data['sampleset'], name=id_merged)
                        f_d_info['parametersets/time_history'][id_merged][...] = f_s_info[f'parametersets_{sub_id}/parametersets/time_history'][_id][...]
                        f_d_info['parametersets/time_history'][id_merged].attrs.modify('finished', True)


except Exception as e:
    print('  WARNING: could not merge')
    print(e)
    pass