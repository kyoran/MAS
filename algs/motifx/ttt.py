

import numpy as np

from algs.motifx.motifx import MotifX

A = np.array([
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
])
print(A)
D = np.diag(np.sum(A, axis=0))
print(D)

M = MotifX(A).M4()
# print(M)
# print(type(M.todense()))
# print(M.todense())
#
# print(type(M.toarray()))
print(M.toarray())

M = M.toarray().astype(np.float32)
M_r = np.reciprocal(M)
M_r[M_r==np.inf] = 0
print(M_r)
