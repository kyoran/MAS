
import random

from algs.convex_minimize import calc_CS
from algs.vis import *
from algs.detect_nei import detect_nei
from algs.union_find import *
from algs.log_each_step import log_each_step

from constants import *

def get_groups(A):
    node_lst = list(range(len(A)))
    edge_lst = []
    for i in node_lst:
        for j in node_lst:  # 结尾点
            # if i != j and j in agent_objs[i].workfor:
            if i != j and A[j, i] == 1:
                edge_lst.append([i, j])

    # 初始化前导节点 pre 数据结构
    pre = [i for i in range(len(node_lst))]
    # 遍历边集合，合并
    for e_i in range(len(edge_lst)):
        edge = edge_lst[e_i]
        join(
            node_lst.index(edge[0]),
            node_lst.index(edge[1]),
            pre
        )
    groups = []
    for n_i in range(len(node_lst)):
        groups.append(find(n_i, pre))
    groups = np.array(groups)

    return groups

def dot_product_angle(v1, v2):
    # 求两个向量的夹角
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0

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


def evolve(points, r_c, r_m, results_path=None, k=None, ax=None, d=0.25):
    agent_num = len(points)
    h = 1 / agent_num  # gaining parameter
    next_points = points.copy()

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

        # print(N_i_Q)
        # print(distances)
        # SDB-Step2
        N_i_Q_head = [] # 里面每个列表存一个象限中距离自己最远的邻居编号
        for one_sec in range(4):
            if len(distances[one_sec]) > 0:
                max_dis = np.max(distances[one_sec])
                farthest_nei_idx = np.where(
                    np.abs(distances[one_sec] - max_dis) < 1e-2     # 允许一定误差范围内的都是最远邻居，增加算法的优势
                )[0]
                tmp_farthest_nei = []
                for kkk in farthest_nei_idx:
                    tmp_farthest_nei.append(N_i_Q[one_sec][kkk])
                N_i_Q_head.append(tmp_farthest_nei)
            else:
                N_i_Q_head.append([])
        # print(N_i_Q_head)

        # SDB-Step3
        T_i_Q = []
        for one_sec in range(4):
            if len(N_i_Q[one_sec]) > 0:
                T_i_Q.append(1)
            else:
                T_i_Q.append(0)
        T_i_Q = np.array(T_i_Q)
        # print(T_i_Q)

        T_i_norm_square = np.around(np.linalg.norm(T_i_Q) ** 2)  # 这边可以求和，也可以这样写
        e_vec = np.array([1, -1, 1, -1])
        # 这边计算是为了和论文保持一样
        N_i_head = []
        if T_i_norm_square != 2 or (T_i_Q * e_vec).sum() != 0:
            for one_sec in range(4):
                if len(N_i_Q_head[one_sec]) > 0:    # 有邻居才随机选
                    N_i_head.append(
                        random.choice(N_i_Q_head[one_sec])
                    )
        else:
            # print("!!!")
            # print("T_i_Q:", T_i_Q)
            # print("N_i_Q_head:", N_i_Q_head)

            m1 = np.where(T_i_Q == 1)[0][0]
            m2 = np.where(T_i_Q == 1)[0][1]

            m1 = N_i_Q_head[m1]
            m2 = N_i_Q_head[m2]

            selected_m1, selected_m2 = None, None
            max_degree = -1

            for one_m1 in m1:
                for one_m2 in m2:
                    # i_location = points[i]
                    one_m1_pos = points[one_m1]
                    one_m2_pos = points[one_m2]
                    m1_i_vec = one_m1_pos * 2 - i_location  # *2 是为了扩大向量的长度，防止因为误差，计算出0向量，这样就不好计算夹角了
                    m2_i_vec = one_m2_pos * 2 - i_location
                    tmp_degree = dot_product_angle(m1_i_vec, m2_i_vec)
                    # print("\ttmp_degree:", tmp_degree)
                    if tmp_degree > max_degree:
                        max_degree = tmp_degree
                        selected_m1 = one_m1
                        selected_m2 = one_m2

            N_i_head.append(selected_m1)
            N_i_head.append(selected_m2)

        # print(N_i_head)

        # SDB-Step4
        u_i_t = np.array([0., 0.])
        if len(neighbors) < agent_num - 1:
            for j in N_i_head:
                d_ij = points[j] - i_location
                u_i_t += d_ij
            u_i_t = u_i_t / len(N_i_head)
        else:
            for j in neighbors:
                d_ij = points[j] - i_location
                u_i_t += d_ij
            u_i_t = u_i_t / len(neighbors)
        # print("u_i_t:", u_i_t)


        ############################################################
        # DSG-Step6
        factor = np.min([
            (r_c-d) / (2*np.linalg.norm(u_i_t)), 1.0
        ])
        u_i_t_head = factor * u_i_t

        # DSG-Step7
        # 求target point
        x_i_head_t = i_location + u_i_t_head

        # DSG-Step8 (Alg. 1)
        A_i_d = np.zeros(shape=(len(neighbors)+1, len(neighbors)+1), dtype=np.float32)
        newidx = list(range(len(neighbors)+1))

        neis = neighbors.copy().tolist()
        neis.insert(0, i)
        nei2idx = dict(zip(neis, newidx))   # {89: 0}
        idx2nei = dict(zip(list(nei2idx.values()), list(nei2idx.keys())))
        for a in neighbors:
            for b in neighbors:
                if a != b and np.linalg.norm(points[a]-points[b]) <= d:
                    A_i_d[nei2idx[a], nei2idx[b]] = 1
                    A_i_d[nei2idx[b], nei2idx[a]] = 1
        for j in neighbors:
            d_ij_norm = np.linalg.norm(points[j] - i_location)
            u_i_t_head_norm = np.linalg.norm(u_i_t_head)
            if d_ij_norm <= (r_c+d)/2 - u_i_t_head_norm:
                A_i_d[nei2idx[i], nei2idx[j]] = 1
                A_i_d[nei2idx[j], nei2idx[i]] = 1
        # print(A_i_d)

        """
        # 可视化测试
        # 画自己
        plt.scatter(i_location[0], i_location[1], color='r')
        # 画邻居
        for j in neighbors:
            plt.scatter(points[j,0], points[j,1], color='b')
        # 画自己和邻居的连边
        for i_nei in np.where(A_i_d[0]==1)[0]:
            plt.plot([i_location[0], points[idx2nei[i_nei], 0]], [i_location[1], points[idx2nei[i_nei], 1]], color='g')
        # 画邻居和邻居的连边
        for jjj in list(range(len(A_i_d)))[1:]:
            for kkk in list(range(len(A_i_d)))[1:]:
                if A_i_d[jjj, kkk] == 1:
                    plt.plot([points[idx2nei[jjj], 0], points[idx2nei[kkk], 0]], [points[idx2nei[jjj], 1], points[idx2nei[kkk], 1]], color='yellow')
        
        plt.savefig(r"E:\papers\2022-TNNLS-HSBMAS\MAS-main\test.jpg")
        """

        # DSG-Step9
        groups = get_groups(A_i_d)      # [4, 1, 4, 4, 4, 6, 6]
        group_id = np.unique(groups)    # [4, 1, 6] # 4是自己的组编号
        min_nei_idxs = []
        for one_group_id in group_id:
            if one_group_id == groups[0]:
                continue
            else:
                one_group_nei_idxs = np.where(groups==one_group_id)[0]  # 找到不连通组里面的邻居下标
                distances = []
                for one_nei_idx in one_group_nei_idxs:
                    one_nei_pos = points[idx2nei[one_nei_idx]]
                    d_ij = np.linalg.norm(one_nei_pos - i_location)
                    distances.append(d_ij)

                min_dis = np.min(distances)
                minest_nei_idx = np.where(
                    np.abs(distances - min_dis) < 1e-2  # 允许一定误差范围内的都是最短邻居，增加算法的优势
                )[0]
                for one_min_nei in one_group_nei_idxs[minest_nei_idx]:
                    min_nei_idxs.append(one_min_nei)

        DP_t = []
        for one_min_nei_idx in min_nei_idxs:
            DP_t.append(idx2nei[one_min_nei_idx])

        # DSG-Step10
        DSG_A = np.zeros(shape=(agent_num, agent_num), dtype=np.float32)
        for j in range(agent_num):
            if j in DP_t:
                DSG_A[i, j] = 1
                DSG_A[j, i] = 1
        try:
            results = calc_CS(x_i_head_t, points, i, DSG_A, r_c, r_m)
        except:
            import pdb; pdb.set_trace()

        next_points[i] = results.x

    # (*) LOG
    if results_path is not None and k is not None:
        L = A.copy()
        row, col = np.diag_indices_from(L)
        L[row, col] = -1. * np.sum(L, axis=1)
        L = -1 * L  # L再取反后，对角线是正数
        log_each_step(results_path, k, A, L)


    if ax is not None:
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)

    return next_points, A
