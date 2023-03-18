from scipy.sparse import csr_matrix, _csparsetools, spmatrix
from scipy.sparse._sparsetools import get_csr_submatrix
from scipy.sparse import coo_matrix, csc_matrix
import numpy as np


class Operator(object):

    # 查找矩阵行切片的非零值, 返回列索引数组
    @staticmethod
    def find_row(rowMat: spmatrix) -> np.ndarray:
        rowMat = coo_matrix(rowMat, copy=False)
        rowMat.sum_duplicates()
        # remove explicit zeros
        nz_mask = rowMat.data != 0
        return rowMat.col[nz_mask]

    # 查找矩阵列切片的非零值, 返回行索引数组
    @staticmethod
    def find_col(colMat: spmatrix) -> np.ndarray:
        colMat = coo_matrix(colMat, copy=False)
        colMat.sum_duplicates()
        # remove explicit zeros
        nz_mask = colMat.data != 0
        return colMat.row[nz_mask]

    # 查找矩阵的非零值, 返回行索引数组和列索引数组
    @staticmethod
    def find(mat: spmatrix) -> tuple:
        mat = coo_matrix(mat, copy=False)
        mat.sum_duplicates()
        # remove explicit zeros
        nz_mask = mat.data != 0
        return mat.row[nz_mask], mat.col[nz_mask]

    # 查找矩阵中大于v的值, 返回行索引数组和列索引数组
    @staticmethod
    def find_gt(mat: spmatrix, v) -> tuple:
        mat = coo_matrix(mat, copy=False)
        mat.sum_duplicates()
        # remove explicit zeros
        nz_mask = mat.data > v
        return mat.row[nz_mask], mat.col[nz_mask]

    # 获取csr矩阵第major行的切片
    @staticmethod
    def csr_row_slice(self: csr_matrix, major: int) -> csr_matrix:
        n = self.shape[0]
        indptr, indices, data = get_csr_submatrix(n, n, self.indptr, self.indices, self.data, major, major + 1, 0, n)
        return self.__class__((data, indices, indptr), shape=(1, n), dtype=self.dtype, copy=False)

    # 获取csc矩阵弟minor列的切片
    @staticmethod
    def csc_col_slice(self: csc_matrix, minor: int) -> csc_matrix:
        n = self.shape[0]
        indptr, indices, data = get_csr_submatrix(n, n, self.indptr, self.indices, self.data, minor, minor + 1, 0, n)
        return self.__class__((data, indices, indptr), shape=(n, 1), dtype=self.dtype, copy=False)

