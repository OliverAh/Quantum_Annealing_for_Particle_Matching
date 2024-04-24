import threading
import queue
import time
import dwave
import dwave.system

import h5py
import numpy as np
import os
import traceback

import sys
import pathlib
sys.path.append(str(pathlib.PurePath(pathlib.Path.cwd().parent)))

from src import h5py_funcs
from src.h5py_funcs import discoveries, init_custom_getstates, io, parameterstudy_using_info_file

class Multithread_Variationstudy:
    '''Simple class that handles the multithreading of, e.g., the parameterstudy using the DWaveSampler. 
    First it collects all the necessary data and information needed. Then starts multiple threads to do the work, and a single thread to write the results to a file.
    For the interaction with DWave annealers this is beneficial, because the overhead of communication to the Annealer is significant. 
    The number of threads for writing the answers to file is always 1, because writing to h5py file is not thread safe (currently at least).
    To specify what each of the threads (workers and writers) do, overload the methods -submitter_work- and -writer_work-, using the following signatures:
        submitter_work(single instance of study, verbose:int=0, print_prefix:str='') -> single solution to write to file
        writer_work: (single solution to write to file, verbose:int=0,print_prefix:str='') -> None
    '''

    def __init__(self, num_threads_submitters:int=5, num_threads_writers:int=1, max_waiting_answers:int=100, method:str='DWave-Annealer'):
        
        self.problems = None

        self.num_threads_submitters = num_threads_submitters
        self.num_threads_writers = num_threads_writers
        self.num_threads_reporters = 1

        self._max_waiting_answers = max_waiting_answers

        self._list_threads_submitters = []
        self._list_threads_writers = []
        self._list_threads_reporters = []

        self.solver = {
        #    'token' : None,
        #    'client' : None,    # e.g. 'qpu'
        #    'architecture' : None,    # e.g. 'pegasus'
        #    'region' : None,    # e.g. 'eu-central-1'
        #    'name' : None     # e.g. 'Advantage_system5.4'
        }

        self.info_folder_path = None
        self.data_folder_path = None
        self.info_file_name = None
        self.data_file_name = None
        self.folder_path_main = None
        # Create the queues for the problems to submit and the answers to write.
        self._queue_problems_to_submit = queue.Queue() 
        self._queue_answers_to_write = queue.Queue(maxsize=self._max_waiting_answers)

        # Create the threading.Event() and threading.Barrier() objects to control the threads.
        #     events are just boolean flags, here used to determine how long the threads should keep trying to retrieve work packages from the queues, used for both -submitters- and -writers-.
        #     barriers are used to make sure that all threads of interest come together at a certain position of the code at the same time, used for -submitters- only, because number of writers is required to be 1 anyways.
        # threading.Event() is boolean, initially False, .set() to True, .clear() to False.
        self._event_flag_submitters_should_work = threading.Event()
        self._event_flag_writers_should_work = threading.Event()
        self._barrier_submitters = threading.Barrier(parties=self.num_threads_submitters, action=self._event_flag_submitters_should_work.clear, timeout=None)
        self._barrier_writers = threading.Barrier(parties=self.num_threads_writers, action=self._event_flag_writers_should_work.clear, timeout=None)
        self._lock_info_file = threading.Lock()
        self._lock_submitters = threading.Lock()
        self._lock_writers = threading.Lock()

        
    def _submitter(self, verbose=0, print_prefix=''):
        mydata_local = threading.local()
        mydata_local.problem = None
        mydata_local.failed_runs = []
        while self._event_flag_submitters_should_work.is_set():
            if not self._queue_problems_to_submit.empty():
                try:
                    mydata_local.problem = self._queue_problems_to_submit.get()
                    if verbose > 0: print(print_prefix + f'submitter {threading.current_thread().name} got problem: {mydata_local.problem}')
                    #time.sleep(1)
                    mydata_local.answer = self.submitter_work(self, mydata_local.problem, verbose=verbose, print_prefix=print_prefix)
                    self._queue_answers_to_write.put(mydata_local.answer)
                    self._queue_problems_to_submit.task_done()
                except Exception as e:
                    mydata_local.answer = e
                    mydata_local.failed_runs.append(mydata_local.problem)
                    print(print_prefix + 'submitter exception, queue was not empty but problem or answer could not be retrieved. Will wait for at least 0.1 seconds and try again.')
                    print(print_prefix + '----', repr(e))
                    time.sleep(0.1)
            elif self._queue_problems_to_submit.empty():
                print(print_prefix + 'submitter queue is empty, will wait at barrier for all submitters to finish.')
                self._barrier_submitters.wait()
                del mydata_local
                return
            else:
                print(print_prefix + 'queue_problems_to_submit is neither -empty- nor -not empty-, this should not happen. Will wait for at least 0.1 seconds before checking again.')
                time.sleep(0.1)


    def _writer(self, verbose=0, print_prefix=''):
        #if self.num_threads_writers != 1:
        #    self._event_flag_submitters_should_work.clear()
        #    print(print_prefix + f'num_threads_writers must always be 1 (currently {self.num_threads_writers}), because writing to h5py file is not thread safe. \n \
        #          One of the writer threads will try to finish the queue_answers_to_write, but for the sake of your own sanity, fix the number of writer threads to 1. :)')

        #    self._lock_multiple_writers.acquire()
        #
        #    if self._lock_multiple_writers.locked():
        #        pass
        #    else:
        #        self._barrier_writers.wait()
        #        return
        mydata_local = threading.local()
        while self._event_flag_writers_should_work.is_set():
            if not self._queue_answers_to_write.empty():
                try:
                    mydata_local.answer = self._queue_answers_to_write.get()
                    if verbose > 0: print(print_prefix + f'writer {threading.current_thread().name} got answer: {mydata_local.answer}')
                    #time.sleep(1)
                    self.writer_work(self, mydata_local.answer, verbose=verbose, print_prefix=print_prefix)
                    self._queue_answers_to_write.task_done()
                except Exception as e:
                    print(print_prefix + 'writer exception, queue was not empty but answer could not be retrieved or written to file. Will wait for at least 0.1 seconds and try again.')
                    print(print_prefix + '----', repr(e))
                    time.sleep(0.1)
            elif self._queue_answers_to_write.empty() and self._event_flag_submitters_should_work.is_set():
                if verbose > 0: print(print_prefix + 'writer queue is empty, will wait for at least 0.1 seconds before checking again.')
                time.sleep(0.1)
            elif self._queue_answers_to_write.empty() and not self._event_flag_submitters_should_work.is_set():
                print(print_prefix + 'writer queue is empty and event_flag_submitters_should_work is not set (all submitters should have finished by now), so writer will also finish.')
                self._barrier_writers.wait()
                del mydata_local
                return
            else:
                print(print_prefix + 'queue_answers_to_write is neither empty nor not empty, this should not happen. Will wait for at least 0.1 seconds before checking again.')
                time.sleep(0.1)

        return # redundant 
    
    #def submitter_work(self, problem, verbose=0):
    #    if verbose > 0: print(f'    problem {problem} is being worked on by {threading.current_thread().name}')
    #    return f'solution to problem {problem}'


    #rng = np.random.default_rng()
    #def submitter_work(problem):
    #    print(f'    problem {problem} is being worked on by {threading.current_thread().name}')
    #    a = rng.random((1000,1000))
    #    b = rng.random((1000,1000))
    #    c = np.dot(a,b).sum()
    #    del a
    #    del b
    #   return f'solution to problem {problem} = {c}'

    #def writer_work(self, answer, verbose=0):
    #    if verbose > 0: print(f'    answer {answer} is being worked on by {threading.current_thread().name}')
    #    return

    def _populate_submitter_queue(self, verbose=0, print_prefix=''):
        #if isinstance(self.problems, list):
        #    for problem in self.problems:
        #        self._queue_problems_to_submit.put(problem)
        #elif isinstance(self.problems, int):
        #    print('ToDo: populating queue with range of problems')
        #else:
        #    print('Could not determine workpackages. Currently list of problems is supported.')
        try: 
            #_ = iter(self.problems)
            for problem in self.problems:
                self._queue_problems_to_submit.put(problem)
        except TypeError:
            raise TypeError('Could not determine workpackages. self.problems must be iterable.')
        return


    def _print_queue_sizes(self, verbose=-1, print_prefix=''):
        mydata_local = threading.local()
        if verbose < 0: print()
        while self._event_flag_writers_should_work.is_set() or self._event_flag_submitters_should_work.is_set():
            if verbose < 0:
                mydata_local.text = 'Queue sizes: problems_to_submit = {}, answers_to_write = {}'.format(self._queue_problems_to_submit.qsize(), self._queue_answers_to_write.qsize())
                print(print_prefix + mydata_local.text, end='\r',)
            if verbose >= 0:
                mydata_local.text = 'Queue sizes: problems_to_submit = {}, answers_to_write = {}'.format(self._queue_problems_to_submit.qsize(), self._queue_answers_to_write.qsize())
                print(print_prefix + mydata_local.text)
            time.sleep(1)
        if verbose < 0: print()


    def _execute(self, verbose=0, print_prefix=''):
        self._event_flag_submitters_should_work.set()
        self._event_flag_writers_should_work.set()

        for i in range(self.num_threads_submitters):
            self._list_threads_submitters.append(threading.Thread(group=None, target=self._submitter, name='Thread_submittter_{:03}'.format(i), args=(), kwargs={'verbose': verbose, 'print_prefix': print_prefix}, daemon=None))
        for i in range(self.num_threads_writers):
            self._list_threads_writers.append(threading.Thread(group=None, target=self._writer, name='Thread_writer_{:03}'.format(i), args=(), kwargs={'verbose' : verbose, 'print_prefix': print_prefix}, daemon=None))
        for i in range(self.num_threads_reporters):
            self._list_threads_reporters.append(threading.Thread(group=None, target=self._print_queue_sizes, name='Thread_reporter_{:03}'.format(i), args=(), kwargs={'verbose' : verbose, 'print_prefix': print_prefix}, daemon=None))

        for t in self._list_threads_submitters:
            print(print_prefix + f'starting thread {t.name}')
            t.start()
        for t in self._list_threads_writers:
            print(print_prefix + f'starting thread {t.name}')
            t.start()
        for t in self._list_threads_reporters:
            print(print_prefix + f'starting thread {t.name}')
            t.start()

        for t in self._list_threads_submitters:
            t.join()
        print(print_prefix + 'Joined all submitter threads.')
        for t in self._list_threads_writers:
            t.join()
        print(print_prefix + 'Joined all writer threads.')
        for t in self._list_threads_reporters:
            t.join()

        print(print_prefix + 'Is queue_problems_to_submit empty:', self._queue_problems_to_submit.empty())
        print(print_prefix + 'Is queue_answers_to_write empty:', self._queue_answers_to_write.empty())

        return 0

    def _is_sane_input(self, print_prefix=''):
        assert 1 <= self.num_threads_submitters, print_prefix + 'num_threads_submitters must be at least 1.'
        #assert 1 == self.num_threads_writers, print_prefix + 'num_threads_writers must always be 1, because writing to h5py file is not thread safe. Maybe that changes in the future.'
        assert dwave.cloud.client.Client.from_config(**self.solver).get_solver().online, print_prefix + 'Could not connect to the solver. Either the solver is offline or the specs are invalid.'
        return True


    def start_execution(self, verbose=0, print_prefix=''):
        assert self._is_sane_input(print_prefix=print_prefix+' '), print_prefix + 'Input is not valid.'
        #self.dwave_client = dwave.cloud.client.Client.from_config(**self.solver)
        self.dwave_client = dwave.cloud.client.Client.from_config(**self.solver)
        self.dwave_solver = self.dwave_client.get_solver(name=self.solver['name'])
        self.dwave_solver_properties = self.dwave_solver.properties
        self.dwave_solver_parameters = self.dwave_solver.parameters
        print(print_prefix + 'Available solvers based on specifications (client): ', self.dwave_client.get_solvers())
        print(print_prefix + 'Chosen solver based on specifications (solver): ', self.dwave_solver, ', status is online: ', self.dwave_solver.online)
        print(print_prefix + '    Solver properties keys: ', self.dwave_solver_properties.keys())
        print(print_prefix + '        problem_run_duration_range: ', self.dwave_solver_properties['problem_run_duration_range'])
        print(print_prefix + '    Solver parameters keys: ', self.dwave_solver_parameters.keys())
        print(print_prefix + '    Solver properties parameters keys: ', self.dwave_solver_properties['parameters'].keys())

        self._populate_submitter_queue(verbose=verbose, print_prefix=print_prefix+' ')
        print(print_prefix + 'Execution started.')
        self._execute(verbose=verbose, print_prefix=print_prefix+' ')
        print(print_prefix + 'Execution finished.')



        # StructuredSolver.check_problem(...)


def init_info_file_parameterstudy_from_dict(dict_info_file):
    h5py_funcs.parameterstudy_using_info_file.prepare_info_file(**dict_info_file)