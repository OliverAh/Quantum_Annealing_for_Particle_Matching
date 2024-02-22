import numpy as np
import scipy as sp

def are_equal_dense_matrices(A,B):
    f_name = 'compare_dense_matrices(A,B)'
    # Check dimensions
    if A.shape != B.shape:
        print('    Msg. in {}: Shapes are different, A:{}, B:{}'.format(f_name, A.shape, B.shape))
        return False
    # Check datatypes
    if A.dtype != B.dtype:
        print('    Msg. in {}: Data types are different, A:{}, B:{}'.format(f_name, A.dtype, B.dtype))
        return False
    # Check equality
    if np.all(A == B):
        return True
    elif np.allclose(A,B):
        print('    Msg. in {}: Equality is False, but Allclose is True'.format(f_name))
        return True


def are_equal_sparse_matrices(A,B):
    f_name = 'compare_sparse_matrices(A,B)'
    # Check dimensions
    if A.shape != B.shape:
        print('    Msg. in {}: Shapes are different, A:{}, B:{}'.format(f_name, A.shape, B.shape))
        return False
    # Check nnz (number of non-zero elements)
    if A.nnz != B.nnz:
        print('    Msg. in {}: Number of non-zero elements (nnz includes explicit zeros) are different, A:{}, B:{}'.format(f_name, A.nnz, B.nnz))
        if A.size != B.size:
            print('    Msg. in {}: Number of non-zero elements are different, A:{}, B:{}'.format(f_name, A.size, B.size))
            return False
        else:
            print('    Msg. in {}: Comparison for .nnz returned False, but for .size returned True, so check specifying explicit zeros. This does not return False for comparison of the matrices.'.format(f_name))
    # Iterate over non-zero elements and compare
    if A.__class__ != B.__class__: # This is a check for the class of the sparse matrix, e.g. csr_matrix, csc_matrix, etc.
        print('    Msg. in {}: Classes are different, as long as both are from the scipy.sparse package that is not an issue. A:{}, B:{}'.format(f_name, A.__class__, B.__class__))
    if not isinstance(A, sp.sparse._csr.csr_array):
        A = A.tocsr()
    if not isinstance(B, sp.sparse._csr.csr_array):
        B = B.tocsr()
    return_marker = True
    if not np.array_equal(A.data,B.data):
        print('    Msg. in {}: Data (non-zero elements) are different, A:{}, B:{}'.format(f_name, A.data, B.data))
        return_marker = False
    if not np.array_equal(A.indices,B.indices):
        print('    Msg. in {}: Indices are different,, A:{}, B:{}'.format(f_name, A.indices, B.indices))
        return_marker = False
    if not np.array_equal(A.indptr,B.indptr):
        print('    Msg. in {}: Pointers are different,, A:{}, B:{}'.format(f_name, A.indptr, B.indptr))
        return_marker = False
    return return_marker
    