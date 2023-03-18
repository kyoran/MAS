
import random

from algs.convex_minimize import calc_CS
from algs.vis import *
from algs.detect_nei import detect_nei
from algs.log_each_step import log_each_step

def which_sector(sec_ranges, one_rotate_base_vec, ij_vector):
    # 邻居j所在i扇区分布内的第几个扇区
    dot = one_rotate_base_vec.dot(ij_vector)  # dot product between [x1, y1] and [x2, y2]
    det = one_rotate_base_vec[0] * ij_vector[1] - one_rotate_base_vec[1] * ij_vector[0]  # determinant
    one_rad = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    one_deg = np.rad2deg(one_rad)   # 这个算出来的范围是1,2象限0-180, 3,4象限-180-0°
    # 转换为0-360范围
    if one_deg < 0:
        one_deg = 360 + one_deg
    one_sec = 3
    for i in range(4):
        if one_deg > sec_ranges[i]:
            one_sec = i
    return one_deg, one_sec

base_vec = np.array([1, 0])
base_sector_ranges = np.array([0, 90, 180, 270])


def evolve(points, r_c, r_m, results_path=None, k=None, ax=None):
    agent_num = len(points)
    h = 1 / agent_num  # gaining parameter

    # (i) 根据感知半径获取所有邻居
    A, L = detect_nei(agent_num, points, r_c)

    # (ii)
    for i in range(agent_num):

        i_location = points[i]
        neighbors = np.where(A[i] == 1)[0]

        # SDB-Step1
        N_i_Q = [[] for _ in range(4)]      # 里面每个列表存一个象限的邻居
        distances = [[] for _ in range(4)]  # 里面每个列表存一个象限的邻居距离自己的距离
        for j in neighbors:
            ij_vector = points[j] - i_location  # 邻居j与i组成的向量
            one_deg, one_sec = which_sector(base_sector_ranges, base_vec, ij_vector)
            N_i_Q[one_sec].append(j)
            # 存邻居距离自己的距离
            ij_dis = np.linalg.norm(points[i] - points[j])
            distances[one_sec].append(ij_dis)


        # SDB-Step2
        N_i_Q_head = [] # 里面每个列表存一个象限中距离自己最远的邻居编号
        for one_sec in range(4):
            max_dis = np.max(distances[one_sec])
            farthest_nei_idx = np.where(
                np.abs(distances[one_sec] - max_dis) < 1e-2     # 允许一定误差范围内的都是最远邻居，增加算法的优势
            )[0]
            N_i_Q_head.append(farthest_nei_idx.tolist())


        # SDB-Step3
        T_i_Q = []
        for one_sec in range(4):
            if len(N_i_Q[one_sec]) > 0:
                T_i_Q[one_sec].append(1)
            else:
                T_i_Q[one_sec].append(0)
        T_i_Q = np.array(T_i_Q)

        T_i_norm_square = np.around(np.linalg.norm(np.array([0, 0, 1, 1])) ** 2)  # 这边可以求和，也可以这样写
        e_vec = np.array([1, -1, 1, -1])
        # 这边计算是为了和论文保持一样
        N_i_head = []
        if T_i_norm_square != 2 or (T_i_Q * e_vec).sum() != 0:
            for one_sec in range(4):
                N_i_head.append(
                    random.choice(N_i_Q_head[one_sec])
                )
        else:
            print("in")
            print(N_i_Q_head)