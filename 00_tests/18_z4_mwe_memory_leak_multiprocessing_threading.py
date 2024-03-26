import sys
import pathlib
sys.path.append(str(pathlib.PurePath(pathlib.Path.cwd().parent)))

import numpy as np
import multiprocessing

#import dwave.system
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

from src import leap_funcs
from src.leap_funcs import qubo
from src.leap_funcs.qubo import parameterstudy
from src.leap_funcs.qubo import test_function

if __name__ == '__main__': # __name__ ist not equal to multiprocessing.current_process().name
    print('Main process started')
    print('__name__', __name__)
    print('active children', multiprocessing.active_children())
    print('current process', multiprocessing.current_process())
    print('current process name', multiprocessing.current_process().name)
    print('current process parrent', multiprocessing.parent_process())
    
else:
    print('Child process started')
    print(' __name__', __name__)
    print(' current process', multiprocessing.current_process())
    print(' current process name', multiprocessing.current_process().name)
    print(' current process parent', multiprocessing.parent_process())
    if multiprocessing.parent_process() != None: print(' current process parent name', multiprocessing.parent_process().name)


with open('../API_Token_Dev.txt', mode='rt') as file:
    token = file.readline().rstrip()
kwargs_dwavesampler = {'token': token, 'architecture': 'pegasus', 'region': 'eu-central-1'}

def overloaded_submitter_work(self, problem, verbose=0, print_prefix=''):
    print(print_prefix + 'start working on problem {}', problem)
    answer = None
    sampler = DWaveSampler(**self.solver)
    if problem == 42:
        raise NotImplementedError('for the purpose of testing, problem {} raised an exception. Let us see what happens'.format(problem))
    #composite = FixedEmbeddingComposite(child_sampler=sampler, embedding=embedding)
    answer = '42 is probably not the answer to problem {}'.format(problem)
    print(print_prefix + 'succesfully retrieved answer for problem {}', problem)
    return answer
    
def overloaded_writer_work(self, answer, verbose=0, print_prefix=''):
        print(print_prefix + 'answer', answer)
        return

def child_process_target(**kwargs):
    print_prefix = kwargs['target']['print_prefix']
    cpn = multiprocessing.current_process().name
    print(print_prefix + cpn, 'started with inputs', kwargs['target'])
    #answer = overloaded_submitter_work(*kwargs['submitter']['args'], **kwargs['submitter']['kwargs'])
    #overloaded_writer_work(*((answer,) + kwargs['writer']['args']), **kwargs['writer']['kwargs'])

    st = leap_funcs.qubo.parameterstudy.Multithread_Variationstudy()
    st.submitter_work = overloaded_submitter_work
    st.writer_work = overloaded_writer_work

    st.problems = kwargs['submitter']['args'][0]
    st.solver.update(**kwargs['target']['kwargs_dwavesampler'])
    st.start_execution(verbose=0)


if __name__ == '__main__':
    '''This multiprocessing implementation is only there to hack around the problem of accumulating memory by repeated instantiations of dwave.system.DWaveSampler.
    The main process splits the whole parameterstudy into chunks of a user-specified size. In a loop over all the chunks, it instantiates a child process using the 'spwan' method, and directly calls .join() after .start().
    Effectively, there is only 1 productive process at a time. The 'benefit' of this is, that the child process can be fully closed (.close()) after having completed its chunk, and the memory is correctly freed (released to the OS).'''
    iterations = 251
    chunk_size = 50
    num_chunks = np.ceil(iterations/chunk_size).astype(int)

    multiprocessing.set_start_method('spawn')
    num_chunks = np.ceil(iterations/chunk_size).astype(int)
    print('num_chunks =', num_chunks)
    print(__name__)
    for chunk_id in range(num_chunks):
        print('chunk', chunk_id+1, 'of', num_chunks)
        ids = np.arange(chunk_id*chunk_size, np.minimum((chunk_id+1)*chunk_size, iterations))
        #print(ids)
        #p = multiprocessing.Process(target=test_function.f_test)
        kwargs_dwavesampler = {'token' : token, 'region':'eu-central-1', 'architecture':'pegasus', 'name':'Advantage_system5.4'}
        kwargs_target = {'print_prefix': ' ', 'kwargs_dwavesampler': kwargs_dwavesampler}
        inputs_submitter = {'args': (ids,), 'kwargs':{'print_prefix': ' '}}
        inputs_writer = {'args': (), 'kwargs':{'print_prefix': ' '}}
        inputs_target = {'args': (), 'kwargs':{'target': kwargs_target, 'submitter': inputs_submitter, 'writer': inputs_writer}}

        p = multiprocessing.Process(target=child_process_target, args=inputs_target['args'], kwargs=inputs_target['kwargs'])
        p.start()
        p.join()
        p.close()