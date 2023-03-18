import numpy as np
import math

from algs.convex_minimize import calc_CS
from algs.vis import *
from algs.detect_nei import detect_nei
from algs.motifx.motifx import MotifX
from algs.log_each_step import log_each_step

from constants import *

def rotate_sector(one_rotate_deg, base_vec):
    one_rotate_rad = np.deg2rad(one_rotate_deg)
    # 旋转后的基准向量
    one_rotate_base_vec = np.array([
        base_vec[0] * np.cos(one_rotate_rad) - base_vec[1] * np.sin(one_rotate_rad),
        base_vec[0] * np.sin(one_rotate_rad) + base_vec[1] * np.cos(one_rotate_rad)
    ])
    return one_rotate_base_vec

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

delta_rotate = 1
base_vec = np.array([1, 0])
base_sector_ranges = np.array([0, 90, 180, 270])

def evolve(points, r_c, r_m, results_path=None, k=None, ax=None):

    next_points = points.copy()
    agent_num = len(points)
    h = 1 / agent_num  # gaining parameter

    # (i) it detects its neighbors according to G;
    A, _ = detect_nei(agent_num, points, r_c)
    C = np.zeros(shape=(agent_num, agent_num), dtype=np.float32)

    # (ii) RSRSP
    if agent_num * (agent_num-1) / 2 != np.sum(A) / 2:  # 不是全连接
        # 每个智能体计算最优的旋转扇区后，选择最近的邻居
        for i in range(agent_num):
            rotate_degs = []
            vars = []

            i_location = points[i]
            neighbors = np.where(A[i] == 1)[0]
            # print(neighbors)
            for one_rotate_deg in range(0, 90, delta_rotate):
                # 基准向量
                # 旋转后的扇区区间范围，旋转后的基准向量
                one_rotate_base_vec = rotate_sector(one_rotate_deg, base_vec)

                sec_num = [0, 0, 0, 0]  # 0,1,2,3扇区中邻居数量
                for j in neighbors:
                    ij_vector = points[j] - i_location  # 邻居j与i组成的向量
                    one_deg, one_sec = which_sector(base_sector_ranges, one_rotate_base_vec, ij_vector)
                    sec_num[one_sec] += 1

                vars.append(np.var(sec_num))
                rotate_degs.append(one_rotate_deg)
                # print(one_rotate_deg, sec_num, np.var(sec_num))
                # plt.show()

            min_idx = np.argmin(vars)
            min_deg = rotate_degs[min_idx]

            # i旋转min_deg后，重新找距离最近的邻居
            nei_idx_in_each_sec = {0: [], 1: [], 2: [], 3: []}  # i的每个扇区中邻居j的编号
            nei_dis_in_each_sec = {0: [], 1: [], 2: [], 3: []}  # i的每个扇区中与邻居j的距离
            for j in neighbors:
                ij_vector = points[j] - i_location  # 邻居j与i组成的向量
                ij_dis = np.linalg.norm(points[i] - points[j])
                one_rotate_base_vec = rotate_sector(min_deg, base_vec)
                one_deg, one_sec = which_sector(base_sector_ranges, one_rotate_base_vec, ij_vector)

                nei_idx_in_each_sec[one_sec].append(j)
                nei_dis_in_each_sec[one_sec].append(ij_dis)

            # print(nei_dis_in_each_sec)
            # print(nei_idx_in_each_sec)
            for one_sec_idx in nei_dis_in_each_sec.keys():
                all_dis = nei_dis_in_each_sec[one_sec_idx]  # 在一个扇区里面选出距离最小的邻居
                # print(type(all_dis), all_dis, len(all_dis))
                if len(all_dis) <= 0:  # 这个扇区没邻居
                    continue
                else:
                    min_dis_idx = np.argmin(all_dis)
                    selected_nei_idx = nei_idx_in_each_sec[one_sec_idx][min_dis_idx]
                    C[i, selected_nei_idx] = 1
                    C[selected_nei_idx, i] = 1

    else:
        C = A.copy()

    # (iii)
    L = C.copy()
    L = -L
    L[np.diag_indices_from(L)] = np.sum(C, axis=0)

    # (*) LOG
    if results_path is not None and k is not None:
        log_each_step(results_path, k, A, L)

    L = np.mat(L)
    points = np.mat(points)
    next_points = (np.mat(np.identity(agent_num)) - h * L) * points
    next_points = np.array(next_points)
    points = np.array(points)

    if ax is not None:
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)

    # 动力学演化
    return next_points, A