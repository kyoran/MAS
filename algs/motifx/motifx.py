from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from .cache import Cache
import numpy as np


class MotifX(object):
    def __init__(self, matrix, reformat=True, dtype=np.int32):
        self.cache = Cache(matrix, reformat, dtype)

    def M1(self) -> (csr_matrix, dict):
        UT_csr: csr_matrix = self.cache.UT_csr
        U_U_csr: csr_matrix = self.cache.U_U_csr
        #   C = (U * U) .* U';
        C: csr_matrix = U_U_csr.multiply(UT_csr)
        #   W = C + C';
        return C + C.transpose()

    def M2(self) -> csr_matrix:
        B_csr: csr_matrix = self.cache.B_csr
        UT_csr: csr_matrix = self.cache.UT_csr
        B_U_csr: csr_matrix = self.cache.B_U_csr
        U_B_csr: csr_matrix = self.cache.U_B_csr
        U_U_csr: csr_matrix = self.cache.U_U_csr
        #   C = (B * U) .* U' + (U * B) .* U' + (U * U) .* B;
        C: csr_matrix = B_U_csr.multiply(UT_csr) + U_B_csr.multiply(UT_csr) + U_U_csr.multiply(B_csr)
        #   W = C + C';
        return C + C.transpose()

    def M3(self) -> csr_matrix:
        B_csr: csr_matrix = self.cache.B_csr
        U_csr: csr_matrix = self.cache.U_csr
        B_B_csr: csr_matrix = self.cache.B_B_csr
        B_U_csr: csr_matrix = self.cache.B_U_csr
        U_B_csr: csr_matrix = self.cache.U_B_csr
        #   C = (B * B) .* U + (B * U) .* B + (U * B) .* B;
        C: csr_matrix = B_B_csr.multiply(U_csr) + B_U_csr.multiply(B_csr) + U_B_csr.multiply(B_csr)
        #   W = C+ C';
        return C + C.transpose()

    def M4(self) -> csr_matrix:
        B_csr: csr_matrix = self.cache.B_csr
        B_B_csr: csr_matrix = self.cache.B_B_csr
        #   W = (B * B) .* B;
        return B_B_csr.multiply(B_csr)

    def M5(self) -> csr_matrix:
        U_csr: csr_matrix = self.cache.U_csr
        U_U_csr: csr_matrix = self.cache.U_U_csr
        UT_U_csr: csr_matrix = self.cache.UT_U_csr
        U_UT_csr: csr_matrix = self.cache.U_UT_csr
        #   T1 = (U  * U ) .* U;
        T1: csr_matrix = U_U_csr.multiply(U_csr)
        #   T2 = (U' * U ) .* U;
        T2: csr_matrix = UT_U_csr.multiply(U_csr)
        #   T3 = (U  * U') .* U;
        T3: csr_matrix = U_UT_csr.multiply(U_csr)
        #   C = T1 + T2 + T3;
        C: csr_matrix = T1 + T2 + T3
        #   W = C + C';
        return C + C.transpose()

    def M6(self) -> csr_matrix:
        B_csr: csr_matrix = self.cache.B_csr
        U_csr: csr_matrix = self.cache.U_csr
        U_B_csr: csr_matrix = self.cache.U_B_csr
        UT_U_csr: csr_matrix = self.cache.UT_U_csr
        #   C1 = (U * B) .* U;
        C1: csr_matrix = U_B_csr.multiply(U_csr)
        #   C1 = C1 + C1';
        C1: csr_matrix = C1 + C1.transpose()
        #   C2 = (U' * U) .* B;
        C2 = UT_U_csr.multiply(B_csr)
        #   W = C1 + C2;
        return C1 + C2

    def M7(self) -> csr_matrix:
        B_csr: csr_matrix = self.cache.B_csr
        UT_csr: csr_matrix = self.cache.UT_csr
        UT_B_csr: csr_matrix = self.cache.UT_B_csr
        U_UT_csr: csr_matrix = self.cache.U_UT_csr
        #   C1 = (U' * B) .* U';
        C1: csr_matrix = UT_B_csr.multiply(UT_csr)
        #   C1 = C1 + C1';
        C1 = C1 + C1.transpose()
        #   C2 = (U * U') .* B;
        C2: csr_matrix = U_UT_csr.multiply(B_csr)
        #   W = C1 + C2;
        return C1 + C2

    def M8(self) -> csr_matrix:
        W_lst = {}
        dtype = self.cache.dtype
        shape = self.cache.shape
        A_dict: dict = self.cache.A_dict
        U_row_find: list = self.cache.U_row_find
        # W = zeros(size(G));
        W: lil_matrix = lil_matrix(shape, dtype=dtype)
        # N = size(G, 1);
        # for i = 1:N
        for i in range(shape[0]):
            # J = find(U(i, :));
            J = U_row_find[i][0]
            # for j1 = 1:length(J)
            for j1 in range(U_row_find[i][1]):
                # for j2 = (j1+1):length(J)
                for j2 in range(j1 + 1, U_row_find[i][1]):
                    # k1 = J(j1);
                    k1 = J[j1]
                    # k2 = J(j2);
                    k2 = J[j2]
                    # if A(k1, k2) == 0 & & A(k2, k1) == 0
                    if not A_dict.get((k1, k2)) and not A_dict.get((k2, k1)):
                        # W(i, k1)  = W(i, k1) + 1;
                        W_lst[(i, k1)] = W_lst.get((i, k1), 0) + 1
                        # W(i, k2)  = W(i, k2) + 1;
                        W_lst[(i, k2)] = W_lst.get((i, k2), 0) + 1
                        # W(k1, k2) = W(k1, k2) + 1;
                        W_lst[(k1, k2)] = W_lst.get((k1, k2), 0) + 1
        row, col, data = [], [], []
        for (i, j), x in W_lst.items():
            row.append(i)
            col.append(j)
            data.append(x)
        W._set_arrayXarray(np.array(row), np.array(col), np.array(data, dtype=dtype))
        #   W = sparse(W + W');
        return W + W.transpose()

    def M9(self) -> csr_matrix:
        W_lst = {}
        dtype = self.cache.dtype
        shape = self.cache.shape
        A_dict: dict = self.cache.A_dict
        U_row_find: list = self.cache.U_row_find
        U_col_find: list = self.cache.U_col_find
        # W = zeros(size(G));
        W: lil_matrix = lil_matrix(shape, dtype=dtype)
        # N = size(G, 1);
        # for i = 1:N
        for i in range(shape[0]):
            # J1 = find(U(i, :));
            J1 = U_row_find[i][0]
            # J2 = find(U(:, i));
            J2 = U_col_find[i][0]
            # for j1 = 1:length(J1)
            for j1 in range(U_row_find[i][1]):
                # for j2 = 1:length(J2)
                for j2 in range(U_col_find[i][1]):
                    # k1 = J1(j1);
                    k1 = J1[j1]
                    # k2 = J2(j2);
                    k2 = J2[j2]
                    # if A(k1, k2) == 0 & & A(k2, k1) == 0
                    if not A_dict.get((k1, k2)) and not A_dict.get((k2, k1)):
                        # W(i, k1)  = W(i, k1) + 1;
                        W_lst[(i, k1)] = W_lst.get((i, k1), 0) + 1
                        # W(i, k2)  = W(i, k2) + 1;
                        W_lst[(i, k2)] = W_lst.get((i, k2), 0) + 1
                        # W(k1, k2) = W(k1, k2) + 1;
                        W_lst[(k1, k2)] = W_lst.get((k1, k2), 0) + 1
        row, col, data = [], [], []
        for (i, j), x in W_lst.items():
            row.append(i)
            col.append(j)
            data.append(x)
        W._set_arrayXarray(np.array(row), np.array(col), np.array(data, dtype=dtype))
        #   W = sparse(W + W');
        return W + W.transpose()

    def M10(self) -> csr_matrix:
        W_lst = {}
        dtype = self.cache.dtype
        shape = self.cache.shape
        A_dict: dict = self.cache.A_dict
        U_row_find_AT: list = self.cache.U_row_find_AT
        # W = zeros(size(G));
        W: lil_matrix = lil_matrix(shape, dtype=dtype)
        # N = size(G, 1);
        # for i = 1:N
        for i in range(shape[0]):
            # J = find(U(i, :));
            J = U_row_find_AT[i][0]
            # for j1 = 1:length(J)
            for j1 in range(U_row_find_AT[i][1]):
                # for j2 = (j1+1):length(J)
                for j2 in range(j1 + 1, U_row_find_AT[i][1]):
                    # k1 = J(j1);
                    k1 = J[j1]
                    # k2 = J(j2);
                    k2 = J[j2]
                    # if A(k1, k2) == 0 && A(k2, k1) == 0
                    if not A_dict.get((k1, k2)) and not A_dict.get((k2, k1)):
                        # W(i, k1)  = W(i, k1) + 1;
                        W_lst[(i, k1)] = W_lst.get((i, k1), 0) + 1
                        # W(i, k2)  = W(i, k2) + 1;
                        W_lst[(i, k2)] = W_lst.get((i, k2), 0) + 1
                        # W(k1, k2) = W(k1, k2) + 1;
                        W_lst[(k1, k2)] = W_lst.get((k1, k2), 0) + 1
        row, col, data = [], [], []
        for (i, j), x in W_lst.items():
            row.append(i)
            col.append(j)
            data.append(x)
        W._set_arrayXarray(np.array(row), np.array(col), np.array(data, dtype=dtype))
        #   W = sparse(W + W');
        return W + W.transpose()

    def M11(self) -> csr_matrix:
        W_lst = {}
        dtype = self.cache.dtype
        shape = self.cache.shape
        A_dict: dict = self.cache.A_dict
        B_row_find: list = self.cache.B_row_find
        U_row_find: list = self.cache.U_row_find
        # W = zeros(size(G));
        W: lil_matrix = lil_matrix(shape, dtype=dtype)
        # N = size(G, 1);
        # for i = 1:N
        for i in range(shape[0]):
            # J1 = find(B(i, :));
            J1 = B_row_find[i][0]
            # J2 = find(U(i, :));
            J2 = U_row_find[i][0]
            # for j1 = 1:length(J1)
            for j1 in range(B_row_find[i][1]):
                # for j2 = 1:length(J2)
                for j2 in range(U_row_find[i][1]):
                    # k1 = J1(j1);
                    k1 = J1[j1]
                    # k2 = J2(j2);
                    k2 = J2[j2]
                    # if A(k1, k2) == 0 && A(k2, k1) == 0
                    if not A_dict.get((k1, k2)) and not A_dict.get((k2, k1)):
                        # W(i, k1)  = W(i, k1) + 1;
                        W_lst[(i, k1)] = W_lst.get((i, k1), 0) + 1
                        # W(i, k2)  = W(i, k2) + 1;
                        W_lst[(i, k2)] = W_lst.get((i, k2), 0) + 1
                        # W(k1, k2) = W(k1, k2) + 1;
                        W_lst[(k1, k2)] = W_lst.get((k1, k2), 0) + 1
        row, col, data = [], [], []
        for (i, j), x in W_lst.items():
            row.append(i)
            col.append(j)
            data.append(x)
        W._set_arrayXarray(np.array(row), np.array(col), np.array(data, dtype=dtype))
        #   W = sparse(W + W');
        return W + W.transpose()

    def M12(self) -> csr_matrix:
        W_lst = {}
        dtype = self.cache.dtype
        shape = self.cache.shape
        A_dict: dict = self.cache.A_dict
        B_row_find: list = self.cache.B_row_find
        U_row_find_AT: list = self.cache.U_row_find_AT
        # W = zeros(size(G));
        W: lil_matrix = lil_matrix(shape, dtype=dtype)
        # N = size(G, 1);
        # for i = 1:N
        for i in range(shape[0]):
            # J1 = find(B(i, :));
            J1 = B_row_find[i][0]
            # J2 = find(U(i, :));
            J2 = U_row_find_AT[i][0]
            # for j1 = 1:length(J1)
            for j1 in range(B_row_find[i][1]):
                # for j2 = 1:length(J2)
                for j2 in range(U_row_find_AT[i][1]):
                    # k1 = J1(j1);
                    k1 = J1[j1]
                    # k2 = J2(j2);
                    k2 = J2[j2]
                    # if A(k1, k2) == 0 && A(k2, k1) == 0
                    if not A_dict.get((k1, k2)) and not A_dict.get((k2, k1)):
                        # W(i, k1)  = W(i, k1) + 1;
                        W_lst[(i, k1)] = W_lst.get((i, k1), 0) + 1
                        # W(i, k2)  = W(i, k2) + 1;
                        W_lst[(i, k2)] = W_lst.get((i, k2), 0) + 1
                        # W(k1, k2) = W(k1, k2) + 1;
                        W_lst[(k1, k2)] = W_lst.get((k1, k2), 0) + 1
        row, col, data = [], [], []
        for (i, j), x in W_lst.items():
            row.append(i)
            col.append(j)
            data.append(x)
        W._set_arrayXarray(np.array(row), np.array(col), np.array(data, dtype=dtype))
        #   W = sparse(W + W');
        return W + W.transpose()

    def M13(self) -> csr_matrix:
        W_lst = {}
        dtype = self.cache.dtype
        shape = self.cache.shape
        A_dict: dict = self.cache.A_dict
        B_row_find: list = self.cache.B_row_find
        # W = zeros(size(G));
        W: lil_matrix = lil_matrix(shape, dtype=dtype)
        # N = size(G, 1);
        # for i = 1:N
        for i in range(shape[0]):
            # J = find(B(i, :));
            J = B_row_find[i][0]
            # for j1 = 1:length(J)
            for j1 in range(B_row_find[i][1]):
                # for j2 = (j1+1):length(J)
                for j2 in range(j1 + 1, B_row_find[i][1]):
                    # k1 = J(j1);
                    k1 = J[j1]
                    # k2 = J(j2);
                    k2 = J[j2]
                    # if A(k1, k2) == 0 && A(k2, k1) == 0
                    if not A_dict.get((k1, k2)) and not A_dict.get((k2, k1)):
                        # W(i, k1)  = W(i, k1) + 1;
                        W_lst[(i, k1)] = W_lst.get((i, k1), 0) + 1
                        # W(i, k2)  = W(i, k2) + 1;
                        W_lst[(i, k2)] = W_lst.get((i, k2), 0) + 1
                        # W(k1, k2) = W(k1, k2) + 1;
                        W_lst[(k1, k2)] = W_lst.get((k1, k2), 0) + 1
        row, col, data = [], [], []
        for (i, j), x in W_lst.items():
            row.append(i)
            col.append(j)
            data.append(x)
        W._set_arrayXarray(np.array(row), np.array(col), np.array(data, dtype=dtype))
        #   W = sparse(W + W');
        return W + W.transpose()

    def Bifan(self) -> csr_matrix:
        W_lst = {}
        dtype = self.cache.dtype
        shape = self.cache.shape
        A_dict: dict = self.cache.A_dict
        U_row_find: list = self.cache.U_row_find
        # W = zeros(size(G));
        W: lil_matrix = lil_matrix(shape, dtype=dtype)
        # NA = ~A & ~A';
        # [ai, aj] = find(triu(NA, 1));
        NA_dict, ai, aj = {}, [], []
        for i in range(0, shape[0]):
            for j in range(i + 1, shape[0]):
                if not A_dict.get((i, j)) and not A_dict.get((j, i)):
                    NA_dict[(i, j)] = 1
                    NA_dict[(j, i)] = 1
                    ai.append(i)
                    aj.append(j)
        # for ind = 1:length(ai)
        for ind in range(len(ai)):
            # x = ai(ind);
            x = ai[ind]
            # y = aj(ind);
            y = aj[ind]
            # xout = find(U(x,:));
            xout = U_row_find[x][0]
            # yout = find(U(y,:));
            yout = U_row_find[y][0]
            # common = intersect(xout, yout);
            common: list = np.intersect1d(xout, yout).tolist()
            # nc = length(common)
            nc = len(common)
            # for i = 1:nc
            for i in range(nc):
                # for j = (i+1):nc
                for j in range(i + 1, nc):
                    # w = common(i);
                    w = common[i]
                    # v = common(j);
                    v = common[j]
                    # if NA(w, v) == 1
                    if NA_dict.get((w, v)):
                        # W(x, y) = W(x, y) + 1;
                        W_lst[(x, y)] = W_lst.get((x, y), 0) + 1
                        # W(x, w) = W(x, w) + 1;
                        W_lst[(x, w)] = W_lst.get((x, w), 0) + 1
                        # W(x, v) = W(x, v) + 1;
                        W_lst[(x, v)] = W_lst.get((x, v), 0) + 1
                        # W(y, w) = W(y, w) + 1;
                        W_lst[(y, w)] = W_lst.get((y, w), 0) + 1
                        # W(y, v) = W(y, v) + 1;
                        W_lst[(y, v)] = W_lst.get((y, v), 0) + 1
                        # W(w, v) = W(w, v) + 1;
                        W_lst[(w, v)] = W_lst.get((w, v), 0) + 1
        row, col, data = [], [], []
        for (i, j), x in W_lst.items():
            row.append(i)
            col.append(j)
            data.append(x)
        W._set_arrayXarray(np.array(row), np.array(col), np.array(data, dtype=dtype))
        #   W = sparse(W + W');
        return W + W.transpose()

    def Edge(self) -> csc_matrix:
        return self.cache.G_csr.copy()
