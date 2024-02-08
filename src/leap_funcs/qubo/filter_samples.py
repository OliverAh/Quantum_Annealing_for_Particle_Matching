import numpy as np

def indices_of_valid_solutions(sample_set, sort_energies=True):
    ids_valid = []
    num_log_qubits = np.shape(sample_set.record[0][0])[0]
    num_particles = int(np.sqrt(num_log_qubits))
    
    for k, sol in enumerate(sample_set.record):
        #print(sol)
        #print(sol[0])
        parts = 0
        poss = 0
        valid_parts = True
        valid_poss = True
        for i in range(num_particles):
            parts = 0
            for j in range(num_particles):
                parts += sol[0][i*num_particles +j]
            #print('parts ', parts)
            if parts != 1:
                valid_parts = False

        for j in range(num_particles):
            poss = 0
            for i in range(num_particles):
                poss += sol[0][i*num_particles +j]
            #print('poss ', poss)
            if poss != 1:
                valid_poss = False
        
        if valid_parts and valid_poss:
            ids_valid.append(k)
    
    #print('ids_valid ', ids_valid)
    
    if sort_energies:
        ids_sorted_energies_full = np.argsort(sample_set.data_vectors['energy'])
        order_of_valid_solutions = np.searchsorted(ids_sorted_energies_full, ids_valid)
        
        return order_of_valid_solutions
    return ids_valid
