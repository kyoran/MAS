from scipy.sparse import spmatrix, lil_matrix, csr_matrix, csc_matrix
from .operator import Operator as op
import numpy as np


# 对重复计算数据进行缓存处理, 以提高多次计算的性能
class Cache(object):
    def __init__(self, matrix, reformat: bool, dtype):

        self.dtype = dtype
        self.shape = None

        self.A_csr = None
        self.__A_dict = None
        self.__AT_csr = None

        self.__B_csr = None
        self.__B_row_find = None

        self.__U_csr = None
        self.__U_csc = None
        self.__UT_csr = None
        self.__U_row_find = None
        self.__U_col_find = None
        self.__U_of_AT_csr = None
        self.__U_row_find_AT = None

        self.__G_csr = None

        self.__B_U_csr = None
        self.__B_B_csr = None

        self.__U_B_csr = None
        self.__U_U_csr = None
        self.__U_UT_csr = None
        self.__UT_U_csr = None
        self.__UT_B_csr = None

        self.formatMatrix(matrix, reformat)

    # 将矩阵转化为csr稀疏矩阵格式
    def formatMatrix(self, matrix, reformat: bool):
        # 重新格式化矩阵, 对角元素设置为0, 所有非0元素设置为1
        if reformat:
            if isinstance(matrix, spmatrix) or isinstance(matrix, np.ndarray):
                A_lil: lil_matrix = lil_matrix(matrix, dtype=self.dtype, copy=True)
            else:
                raise ValueError("only accept scipy.sparse.spmatrix Or numpy.ndarray")
            # A = A - diag(diag(A));
            A_lil.setdiag(0)
            # A(find(A)) = 1
            gt1_find: tuple = op.find_gt(A_lil, 1)
            data = np.ones(gt1_find[0].shape[0], dtype=self.dtype)
            A_lil._set_arrayXarray(gt1_find[0], gt1_find[1], data)
            self.A_csr: csr_matrix = A_lil.tocsr()
        else:
            if isinstance(matrix, spmatrix) or isinstance(matrix, np.ndarray):
                self.A_csr: csr_matrix = csr_matrix(matrix, dtype=self.dtype, copy=True)
            else:
                raise ValueError("only accept scipy.sparse.spmatrix Or numpy.ndarray")

        self.shape = self.A_csr.shape

    # 初始矩阵A的((i, j): v)dict形式, 用于单个元素的读取和行切片
    @property
    def A_dict(self) -> dict:
        if self.__A_dict is None:
            self.__A_dict: dict = dict(self.A_csr.todok().items())
        return self.__A_dict

    # 初始矩阵A的csr稀疏矩阵形式, 用于矩阵加法和矩阵乘法和行切片
    @property
    def AT_csr(self) -> csr_matrix:
        if self.__AT_csr is None:
            self.__AT_csr = self.A_csr.transpose().tocsr()
        return self.__AT_csr

    # 矩阵B的csr稀疏矩阵形式, 用于矩阵加法和矩阵乘法和行切片
    @property
    def B_csr(self):
        if self.__B_csr is None:
            # B = spones(A&A');  % bidirectional
            B_lil: lil_matrix = lil_matrix(self.shape, dtype=self.dtype)
            gt1_find = op.find_gt(self.A_csr + self.AT_csr, 1)
            data = np.ones(gt1_find[0].shape[0], dtype=self.dtype)
            B_lil._set_arrayXarray(gt1_find[0], gt1_find[1], data)
            self.__B_csr: csr_matrix = B_lil.tocsr()
        return self.__B_csr

    # 缓存矩阵B的每行矩阵切片的非零值索引查找
    # find(B_csr[i, :])[1].tolist() == B_rwo_find[i]
    @property
    def B_row_find(self) -> list:
        if self.__B_row_find is None:
            B_row_find = []
            B_csr: csr_matrix = self.B_csr
            for i in range(self.shape[0]):
                lst: list = op.find_row(op.csr_row_slice(B_csr, i)).tolist()
                length = len(lst)
                B_row_find.append((lst, length))
            self.__B_row_find = B_row_find
        return self.__B_row_find

    # 矩阵U的csr稀疏矩阵形式, 用于矩阵加法和矩阵乘法和行切片
    @property
    def U_csr(self) -> csr_matrix:
        if self.__U_csr is None:
            # U = A - B;
            self.__U_csr: csr_matrix = self.A_csr - self.B_csr
            # 释放内存
            if self.__G_csr is not None:
                del self.A_csr
        return self.__U_csr

    # 矩阵U的csc稀疏矩阵形式, 用于矩阵列切片
    @property
    def U_csc(self) -> csc_matrix:
        if self.__U_csc is None:
            self.__U_csc: csc_matrix = self.U_csr.tocsc()
        return self.__U_csc

    # 矩阵U转置矩阵的csr稀疏矩阵形式, 用于矩阵列切片
    @property
    def UT_csr(self) -> csr_matrix:
        if self.__UT_csr is None:
            self.__UT_csr: csr_matrix = self.U_csr.transpose().tocsr()
        return self.__UT_csr

    # 缓存矩阵U的每行矩阵切片的非零值索引查找
    # find(U_csr[i, :])[1].tolist() == U_rwo_find[i]
    @property
    def U_row_find(self) -> list:
        if self.__U_row_find is None:
            U_row_find = []
            U_csr: csr_matrix = self.U_csr
            for i in range(self.shape[0]):
                lst: list = op.find_row(op.csr_row_slice(U_csr, i)).tolist()
                length = len(lst)
                U_row_find.append((lst, length))
            self.__U_row_find = U_row_find
        return self.__U_row_find

    # 缓存矩阵U的每行矩阵切片的非零值索引查找
    # find(U_csc[:, i])[0].tolist() == U_col_find[i]
    @property
    def U_col_find(self) -> list:
        if self.__U_col_find is None:
            U_col_find = []
            U_csc: csc_matrix = self.U_csc
            for i in range(self.shape[0]):
                lst: list = op.find_col(op.csc_col_slice(U_csc, i)).tolist()
                length = len(lst)
                U_col_find.append((lst, length))
            self.__U_col_find = U_col_find
            # 释放内存
            del self.__U_csc
        return self.__U_col_find

    # 由A的转置矩阵AT计算出来的U_of_AT_csr的csr矩阵形式, 用于矩阵列切片
    @property
    def U_of_AT_csr(self) -> csr_matrix:
        if self.__U_of_AT_csr is None:
            self.__U_of_AT_csr: csr_matrix = self.AT_csr - self.B_csr
        return self.__U_of_AT_csr

    # 缓存由A的转置矩阵AT计算出来的U_of_AT_csr矩阵的的非零值索引查找
    # find(U_of_AT_csr[i, :])[1].tolist() == U_row_find_AT[i]
    @property
    def U_row_find_AT(self) -> list:
        if self.__U_row_find_AT is None:
            U_row_find_AT = []
            U_of_AT_csr: csr_matrix = self.U_of_AT_csr
            for i in range(self.shape[0]):
                lst: list = op.find_row(op.csr_row_slice(U_of_AT_csr, i)).tolist()
                length = len(lst)
                U_row_find_AT.append((lst, length))
            self.__U_row_find_AT = U_row_find_AT
            # 释放内存
            del self.__U_of_AT_csr
        return self.__U_row_find_AT

    # 矩阵G的csr稀疏矩阵形式, 用于矩阵加法和矩阵乘法和行切片
    @property
    def G_csr(self) -> csr_matrix:
        if self.__G_csr is None:
            # G = A | A';
            G_csr: lil_matrix = (self.A_csr + self.AT_csr).tolil()
            gt1_find = op.find_gt(G_csr, 1)
            data = np.ones(gt1_find[0].shape[0], dtype=self.dtype)
            G_csr._set_arrayXarray(gt1_find[0], gt1_find[1], data)
            self.__G_csr = G_csr.tocsr()
            # 释放内存
            if self.__U_csr is not None:
                del self.A_csr
        return self.__G_csr

    # B_csr.dot(U_csr)
    @property
    def B_U_csr(self) -> csr_matrix:
        if self.__B_U_csr is None:
            self.__B_U_csr = self.B_csr.dot(self.U_csr)
        return self.__B_U_csr

    # B_csr.dot(B_csr)
    @property
    def B_B_csr(self) -> csr_matrix:
        if self.__B_B_csr is None:
            self.__B_B_csr = self.B_csr.dot(self.B_csr)
        return self.__B_B_csr

    # U_csr.dot(B_csr)
    @property
    def U_B_csr(self) -> csr_matrix:
        if self.__U_B_csr is None:
            self.__U_B_csr = self.U_csr.dot(self.B_csr)
        return self.__U_B_csr

    # U_csr.dot(U_csr)
    @property
    def U_U_csr(self) -> csr_matrix:
        if self.__U_U_csr is None:
            self.__U_U_csr = self.U_csr.dot(self.U_csr)
        return self.__U_U_csr

    # U_csr.dot(UT_csr)
    @property
    def U_UT_csr(self) -> csr_matrix:
        if self.__U_UT_csr is None:
            self.__U_UT_csr = self.U_csr.dot(self.UT_csr)
        return self.__U_UT_csr

    # UT_csr.dot(U_csr)
    @property
    def UT_U_csr(self) -> csr_matrix:
        if self.__UT_U_csr is None:
            self.__UT_U_csr = self.UT_csr.dot(self.U_csr)
        return self.__UT_U_csr

    # UT_csr.dot(B_csr)
    @property
    def UT_B_csr(self) -> csr_matrix:
        if self.__UT_B_csr is None:
            self.__UT_B_csr = self.UT_csr.dot(self.B_csr)
        return self.__UT_B_csr
