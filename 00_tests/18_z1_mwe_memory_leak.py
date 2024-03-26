import gc
import numpy as np

import time

import dwave
import dwave.system
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import minorminer



with open('API_Token_Dev.txt') as file:
    token = file.readline().rstrip()
kwargs_dwavesampler = {'token': token, 'architecture': 'pegasus', 'region': 'eu-central-1'}
sampler = DWaveSampler(**kwargs_dwavesampler)


qubo_q = np.random.rand(5,5)
qubo_q = qubo_q + qubo_q.T
source_graph = {(i+1, j+1): qubo_q[i, j] for i in range(5) for j in range(i,5)}


embedding = minorminer.find_embedding(S=source_graph, T=sampler.edgelist)

sampler = DWaveSampler(token = token, architecture='pegasus', region='eu-central-1')

composite = FixedEmbeddingComposite(child_sampler=sampler, embedding=embedding)

del composite
#del sampler

iterations = 1000
for i in range(iterations):
    print(f'iteration: {i+1} of {iterations}', end='\r')
    #time.sleep(0.5)
    #sampler = DWaveSampler(token = token, architecture='pegasus', region='eu-central-1')
    composite = FixedEmbeddingComposite(child_sampler=sampler, embedding=embedding)
    #a = composite.embedding_parameters
    #if 'hello' in a.keys():
    #    print(a['hello'])
    #del sampler
    del composite
    