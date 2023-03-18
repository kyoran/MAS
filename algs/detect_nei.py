
import numpy as np

def detect_nei(agent_num, points, r_c):
    A = np.zeros(shape=(agent_num, agent_num), dtype=np.float32)
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            # print(np.linalg.norm(points[i]-points[j]) - r_c)
            # print(abs(np.linalg.norm(points[i]-points[j]) - r_c))
            if (np.linalg.norm(points[i]-points[j]) - r_c) <= 1e-2:     # 凸优化那边计算有误差，所以这边要防止断开连接
            # if np.linalg.norm(points[i] - points[j]) <= r_c:
                A[i, j] = 1
                A[j, i] = 1

    L = A.copy()
    row, col = np.diag_indices_from(L)
    L[row, col] = -1. * np.sum(L, axis=1)
    L = -1 * L  # L再取反后，对角线是正数

    return A, L
