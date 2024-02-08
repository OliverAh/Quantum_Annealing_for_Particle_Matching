import dimod
import dwave
def _custom__getstate__dwave_cloud_config_models_ClientConfig(self):
    #print('customized __getstate of dwave.cloud.config.models.ClientConfig')
    state = self.__dict__.copy()
    del state['token']
    state['polling_schedule'] = state['polling_schedule'].__dict__
    state['request_retry'] = state['request_retry'].__dict__
    return state

def _custom__getstate__dwave_cloud_client_qpu_client(self):
    #print('customized __getstate of dwave.cloud.client.qpu.Client')
    state = self.__dict__.copy()
    data_to_remove = ['defaults', '_submission_queue', '_submission_workers',
                      '_cancel_queue', '_cancel_workers', '_poll_queue', 
                      '_poll_workers', '_load_queue', '_load_workers', 
                      '_upload_problem_executor', '_upload_part_executor', 
                      '_encode_problem_executor', 'session']
    for key in data_to_remove:
        del state[key]
    try: 
        del state['token']
    except: 
        try: del state['config']['token']
        except: pass
    state['config'] = state['config'].__getstate__()
    return state

def _custom__getstate__dwave_cloud_solver_StructuredSolver(self):
    #print('customized __getstate of dwave.cloud.solver.StructuredSolver')
    state = self.__dict__.copy()
    for key, value in state.items():
        if isinstance(value, dwave.cloud.client.qpu.Client):
            state[key] = value.__getstate__()
    return state


def _custom__getstate__dwave_system_samplers_dwave_samplers_DWaveSampler(self):
    #print('customized __getstate of dwave.system.samplers.dwave_sampler.DWaveSampler')
    state = self.__dict__.copy()
    for key, value in state.items():
        if isinstance(value, dwave.cloud.client.qpu.Client):
            state[key] = value.__getstate__()
        if isinstance(value, dwave.cloud.solver.StructuredSolver):
            state[key] = value.__getstate__()
    try: 
        state['_adjacency'] = {str(key): list(value) for key, value in state['_adjacency'].items()}
    except: 
        tmp = dwave.system.EmbeddingComposite(self)
        state['_adjacency'] = {str(key): list(value) for key, value in state['_adjacency'].items()}
    return state

def _custom__getstate__dwave_system_composites_embedding_FixedEmbeddingComposite(self):
    #print('customized __getstate of dwave.system.composite.embeddings.FixedEmbeddingComposite')
    state = self.__dict__.copy()
    for key, value in state.items():
        if isinstance(value, dwave.system.samplers.dwave_sampler.DWaveSampler):
            state[key] = value.__getstate__()
        elif isinstance(value, dimod.core.structured._Structure):
            state[key] = value.__getstate__()
        elif key == 'properties':
            try: tmp_chain_strength = state[key]['embedding'].chain_strength
            except: tmp_chain_strength = state[key]['embedding']['chain_strength']
            state[key]['embedding'] = {str(key2): value2 for key2,value2 in state[key]['embedding'].items()}
            state[key]['embedding']['chain_strength'] = tmp_chain_strength
        elif key == 'embedding':
            try: tmp_chain_strength = state[key].chain_strength
            except: tmp_chain_strength = state[key]['chain_strength']
            state[key] = {str(key2): value2 for key2,value2 in state[key].items()}
            state[key]['chain_strength'] = tmp_chain_strength
        elif key == 'children' and isinstance(value, list):
            state[key] = {'child_{}'.format(i): item.__getstate__() for i, item in enumerate(value)}
    if 'find_embedding' in state.keys():
        del state['find_embedding']
    return state

dwave.cloud.config.models.ClientConfig.__getstate__ = _custom__getstate__dwave_cloud_config_models_ClientConfig
dwave.cloud.client.qpu.Client.__getstate__ = _custom__getstate__dwave_cloud_client_qpu_client
dwave.cloud.solver.StructuredSolver.__getstate__ = _custom__getstate__dwave_cloud_solver_StructuredSolver
dwave.system.samplers.dwave_sampler.DWaveSampler.__getstate__ = _custom__getstate__dwave_system_samplers_dwave_samplers_DWaveSampler
dwave.system.composites.embedding.FixedEmbeddingComposite.__getstate__ = _custom__getstate__dwave_system_composites_embedding_FixedEmbeddingComposite

print('Custom getstate functions for dwave.cloud.config.models.ClientConfig, dwave.cloud.client.qpu.Client, dwave.cloud.solver.StructuredSolver, dwave.system.samplers.dwave_sampler.DWaveSampler, dwave.system.composites.embedding.FixedEmbeddingComposite have been initialized.')