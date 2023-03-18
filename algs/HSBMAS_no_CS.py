# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/12 19:58

import time
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from algs.timer import *
from algs.union_find import *
from algs.convex_minimize import calc_CS
from algs.vis import vis_HSBMAS
from algs.log_each_step import log_each_step
from algs.detect_nei import detect_nei
from constants import *

def calc_1_2_hop_connectivity_network(agent_num, points, r_c):

    """计算1-hop和2-hop邻居
            1-hop标注为1
            2-hop标注为2
            """
    C = np.zeros(shape=(agent_num, agent_num), dtype=np.float32)
    # 先计算1-hop邻居
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            dis = np.linalg.norm(
                points[i] - points[j]
            )
            if dis <= r_c:
                C[i, j] = 1
                C[j, i] = 1
                # D[i] += dis

    # 再计算2-hop邻居
    for i in range(agent_num):
        # 找i的2-hop邻居k : i - j - k
        for j in range(agent_num):
            if j == i:
                continue
            if C[i, j] == 1:
                for k in range(agent_num):
                    if k == j or k == i:
                        continue
                    if C[i, k] == 1:  # 已经是1-hop邻居的就不再让他成为2-hop
                        continue
                    else:
                        if C[j, k] == 1:
                            C[i, k] = 2
                            C[k, i] = 2
    return C


def get_nei(idx, C, type=3):
    """
    type=1: 1-hop
    type=2: 2-hop
    type=3: 1-hop和2-hop
    """
    _1_hop_nei_idx = np.where(C[idx] == 1)[0]
    _2_hop_nei_idx = np.where(C[idx] == 2)[0]
    if type == 1:
        return _1_hop_nei_idx
    elif type == 2:
        return _2_hop_nei_idx
    elif type == 3:
        return _1_hop_nei_idx, _2_hop_nei_idx


def random_uniform_in_circle(circle_xy, r_c, samples_num=2000):
    t = np.random.random(size=samples_num) * 2 * np.pi - np.pi
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0, samples_num, 1)
    for i in i_set:
        len = np.sqrt(np.random.random()) * r_c
        x[i] = x[i] * len + circle_xy[0]
        y[i] = y[i] * len + circle_xy[1]
    return np.column_stack((x, y))

def calc_priority(i_1_hop_nei_idx, i_2_hop_nei_idx, i, points, r_c):
    _1_hop_nei_num = len(i_1_hop_nei_idx)
    _2_hop_nei_num = len(i_2_hop_nei_idx)
    i_xy = points[i]

    sample_xys = random_uniform_in_circle(i_xy, r_c, samples_num=priority_sample_num)  # 以智能体i位置为圆心，感知半径为圆采样2000个点
    select_idx = np.zeros(shape=sample_xys.shape[0], dtype=np.int32)
    # 统计，以邻居j为圆心，感知半径为一半的圆与i圆的交集
    for j in i_1_hop_nei_idx:
        j_xy = points[j]
        tmp_dis = np.linalg.norm(sample_xys - j_xy, axis=1)
        select_idx[np.where(tmp_dis <= r_c / 2)[0]] = 1
    select_num = np.sum(select_idx)
    P = select_num / priority_sample_num

    return P + i / 1000    # 跑样例的时候加进去


@timer
def evolve(points, r_c, r_m, results_path=None, time_k=None, ax=None):
    np.random.seed(19961110)

    next_points = points.copy()
    agent_num = len(points)

    # (i) 获取1跳和2跳邻居
    sss_time = time.time()
    C = calc_1_2_hop_connectivity_network(agent_num, points, r_c)
    A, _ = detect_nei(agent_num, points, r_c)
    print(f"\t[Done ({time.time()-sss_time:.3f})s] 获取1跳和2跳邻居信息")

    # (ii) TMPO
    # ii.1 初始化参数
    sss_time = time.time()
    chs = [i for i in range(agent_num)]     # 每个智能体选择自己为ch
    workfors = [[] for i in range(agent_num)]   # 每个下标是一个智能体的workfor
    types = ["NB" for i in range(agent_num)]    # 每个下标是一个智能体的类别
    hashes = [i/1e-5 for i in range(agent_num)] # 每个下标是一个智能体的下标hash值
    _1hop_nei_idxs = []     # 每个下标是一个智能体的1跳邻居下标
    _2hop_nei_idxs = []     # 每个下标是一个智能体的2跳邻居下标
    prioritys = []
    for i in range(agent_num):
        i_1_hop_nei_idx, i_2_hop_nei_idx = get_nei(i, C, 3)
        _1hop_nei_idxs.append(i_1_hop_nei_idx)
        _2hop_nei_idxs.append(i_2_hop_nei_idx)
        prioritys.append(calc_priority(i_1_hop_nei_idx, i_2_hop_nei_idx, i, points, r_c) + hashes[i])
    print(f"\t[Done ({time.time()-sss_time:.3f})s] 初始化TMPO算法的相关参数")

    # ii.2 IS CLUSTERHEAD
    sss_time = time.time()

    for i in range(agent_num):
        chs[i] = i
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        for j in set(i_1_hop_nei_idx) | {i}:      # [!]
            if prioritys[j] >= prioritys[chs[i]]:
                chs[i] = j
    for i in range(agent_num):
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        for j in set(i_1_hop_nei_idx) | {i}:      # [!]
            if chs[j] == i:
                types[i] = "CH"
                break

    print(f"\t[Done ({time.time()-sss_time:.3f})s] IS clusterhead")


    # ii.3 IS DOORWAY
    sss_time = time.time()

    for i in range(agent_num):
        if types[i] == "CH":
            continue

        # 每个智能体判断是不是doorway
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        for j in i_1_hop_nei_idx:   # [i nei: j]
            if types[j] == "CH":
                j_1_hop_nei_idx = set(get_nei(j, C, 1)) - {i}
                for k in i_1_hop_nei_idx:   # [i nei: k] - {j}
                    if k != j and types[k] != "CH":
                        k_1_hop_nei_idx = set(get_nei(k, C, 1)) - {j, i}
                        for n in k_1_hop_nei_idx:   # [k nei: n] - {i, j}
                            if types[n] == "CH" and \
                                    n not in set(i_1_hop_nei_idx) | set(j_1_hop_nei_idx):
                                n_1_hop_nei_idx = get_nei(n, C, 1)
                                # Case a: n, j不是2-hop
                                flag_a = False
                                for m in i_1_hop_nei_idx:  # [m]
                                    if m == j or m == n or m == k:  # [附加!]
                                        continue
                                    m_1_hop_nei_idx = get_nei(m, C, 1)
                                    if j in m_1_hop_nei_idx and n in m_1_hop_nei_idx:
                                        flag_a = True
                                        break
                                if flag_a:
                                    # print("flag_a")
                                    continue

                                # Case b or c:
                                flag_bc = False
                                for m in i_1_hop_nei_idx:
                                    if m == j or m == n or m == k or m == i:  # [附加!]
                                        continue
                                    m_1_hop_nei_idx = get_nei(m, C, 1)
                                    sss = set(i_1_hop_nei_idx) & set(m_1_hop_nei_idx)
                                    sss = sss - {i, m, j, k, n}  # [附加!]
                                    # case b
                                    if len(sss) == 0:
                                        if n in m_1_hop_nei_idx and types[m] == "CH":
                                            flag_bc = True
                                            break
                                    # case c
                                    else:
                                        for p in sss:
                                            if n in m_1_hop_nei_idx and (
                                                    types[m] == "CH" or types[p] == "CH"):
                                                flag_bc = True
                                                break
                                if flag_bc:
                                    # print("flag_bc")
                                    continue

                                # Case d:
                                flag_d = False
                                s1 = set(i_1_hop_nei_idx) & set(n_1_hop_nei_idx)
                                s1 = s1 - {k, j, i, n}  # [附加!]
                                for m in s1:
                                    m_1_hop_nei_idx = get_nei(m, C, 1)
                                    sss = set(j_1_hop_nei_idx) & set(m_1_hop_nei_idx)
                                    sss = sss - {i, j, m, k, n}  # [附加!]
                                    if len(sss) == 0:
                                        if prioritys[m] > prioritys[i]:
                                            flag_d = True
                                            break
                                    else:
                                        for p in sss:
                                            if prioritys[m] > prioritys[i] or \
                                                    prioritys[p] > prioritys[i]:
                                                flag_d = True
                                                break
                                if flag_d:
                                    # print("flag_d")
                                    continue

                                """"""
                                types[i] = "DW"
                                workfors[i].append(j)
                                workfors[i].append(n)
    for i in range(agent_num):
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        if types[i] == "DW":
            delete_workfor = []
            for one_workfor_idx in workfors[i]:
                if one_workfor_idx not in i_1_hop_nei_idx:
                    delete_workfor.append(one_workfor_idx)
            for iii in delete_workfor:
                workfors[i].remove(iii)

    print(f"\t[Done ({time.time()-sss_time:.3f})s] IS doorway")

    # ii.4 IS GATEWAY
    sss_time = time.time()

    for i in range(agent_num):
        if types[i] == "CH":
            continue

        # 每个智能体判断是不是gateway
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        for j in i_1_hop_nei_idx:
            if types[j] == "CH" or types[j] == "DW":    # j为CH或DW
                j_1_hop_nei_idx = get_nei(j, C, 1)
                for k in i_1_hop_nei_idx:
                    if k == j or k == i or j == i:
                        continue
                    if k not in j_1_hop_nei_idx and \
                            (types[k] == "CH" or types[k] == "DW") and \
                            (types[k] != "DW" or types[j] != "DW"): # 只能有一个DW
                        # k为CH或DW
                        # 且j\k不同时为DW

                        k_1_hop_nei_idx = get_nei(k, C, 1)

                        flag = False
                        sss = set(j_1_hop_nei_idx) & set(k_1_hop_nei_idx)
                        sss = sss - {k, j, i}
                        for n in sss:
                            if (types[n] == "CH" or types[n] == "DW") or (prioritys[n] > prioritys[i]):
                                flag = True
                                break
                        if flag:
                            # print("flag")
                            continue
                        else:
                            types[i] = "GW"
                            workfors[i].append(j)
                            workfors[i].append(k)
    print(f"\t[Done ({time.time()-sss_time:.3f})s] IS gateway")


    # iii. refine_workfor
    sss_time = time.time()

    for i in range(agent_num):
        i_1_hop_nei_idx = _1hop_nei_idxs[i]

        if types[i] == "CH":
            """CH与所有1-hop的CH连接"""
            for j in i_1_hop_nei_idx:
                if types[j] == "CH":
                    workfors[i].append(j)
        if types[i] == "NB":
            workfors[i] = [chs[i]]

        workfors[i] = list(set(workfors[i]))

    print(f"\t[Done ({time.time()-sss_time:.3f})s] refine workfor")

    # iv. 剪枝操作
    sss_time = time.time()

    # 首先生成根据主干网络生成的邻接矩阵（初步选择的邻居）
    newC = np.zeros((agent_num, agent_num), dtype=np.int32)  # 0阵
    for i in range(agent_num):
        newC[i, workfors[i]] = 1
    # 再把有向图转为无向图
    newC = newC | newC.T
    # 下面开始剪枝
    # iv.1 剪枝1
    for i in range(agent_num):
        i_1_hop_nei_idx = _1hop_nei_idxs[i]

        if types[i] == "CH":
            # """CH与所有1-hop的CH连接"""
            """优化: 3个CH组成三角形的话，就去掉最低优先级的CH的连接"""
            for j in i_1_hop_nei_idx:
                if types[j] == "CH":
                    for k in i_1_hop_nei_idx:
                        if types[k] == "CH" and k != j:
                            # 三个CH是连成三角形的
                            if newC[j, k] == 1:
                                # 找到i, j、k中优先级最小的那个CH
                                CH_idx_lst = [i, j, k]
                                CH_priority_lst = [prioritys[i], prioritys[j], prioritys[k]]
                                min_priority_CH_id = CH_idx_lst[np.argmin(CH_priority_lst)]
                                max_priority_CH_id = CH_idx_lst[np.argmax(CH_priority_lst)]

                                if min_priority_CH_id == i:
                                    # 最小的是自己，就删掉i与第二小的连接
                                    CH_idx_lst.remove(i)
                                    CH_idx_lst.remove(max_priority_CH_id)
                                    newC[i, CH_idx_lst[0]] = 0
                                    newC[CH_idx_lst[0], i] = 0
    # iv.2 剪枝2
    for i in range(agent_num):
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        if types[i] == "DW":
            """优化: Refine connections between doorways and clusterheads"""
            for j in i_1_hop_nei_idx:
                if types[j] == "CH":
                    for k in i_1_hop_nei_idx:
                        if types[k] == "CH" and k != j:
                            if newC[j, k] == 1:
                                # j,k得相连才行
                                # 找到j、k两个CH，看哪个priority最小
                                min_priority_CH_id = j
                                if prioritys[j] > prioritys[k]:
                                    min_priority_CH_id = k
                                # 去掉
                                if newC[min_priority_CH_id, i]:
                                    newC[i, min_priority_CH_id] = 0
                                    newC[min_priority_CH_id, i] = 0
    # iv.3 剪枝3
    for i in range(agent_num):
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        if types[i] == "GW":
            """【优化】GW/DW与多个CH连接时，只与最高优先级的CH相连"""
            # 使用并查集的思路（https://www.cnblogs.com/zhongzihao/p/9277530.html）
            # k表示分组数量，n表示节点数量，m表示边数量
            # 并查集：最好的情况下为O(m+n)，最差的情况下为O(m*logn+m)
            # 1. 初始化邻居点集合
            node_lst = []
            for j in i_1_hop_nei_idx:
                if types[j] == "CH" and newC[j, i] == 1:
                    node_lst.append(j)

            if len(node_lst) >= 3:
                # 2. 初始化边集合
                edge_lst = []
                for i_node in node_lst:  # 起始点
                    for j_node in node_lst:  # 结尾点
                        if i_node != j_node and newC[j_node, i_node] == 1:
                            edge_lst.append([i_node, j_node])

                # 3. 分组（看哪些组内成员数>=2)
                if len(edge_lst) != 0:
                    # 4. 初始化前导节点 pre 数据结构
                    pre = [i_node for i_node in range(len(node_lst))]
                    # 5. 遍历边集合，合并
                    for e_i in range(len(edge_lst)):
                        edge = edge_lst[e_i]
                        join(
                            node_lst.index(edge[0]),
                            node_lst.index(edge[1]),
                            pre
                        )
                    # 6. 遍历点集合，查看分组
                    groups = []
                    for n_i in range(len(node_lst)):
                        groups.append(find(n_i, pre))
                    groups = np.array(groups)
                    # if len(set(groups)) > 1:    # 与i相连的有两个分组，这2个组不相连
                    node_lst = np.array(node_lst)
                    for k, v in Counter(groups).items():
                        if v >= 2:
                            # 每个类别中超过2个的选出来，然后只保留最高的优先级对应的智能体的连接
                            sss = node_lst[np.where(groups == k)[0]]  # 先把所有相连的1-hop的CH id放进去
                            ppp = [prioritys[aaa] for aaa in sss]  # 放入对应的CH的priority
                            # 最大优先级的那个CH
                            max_priority_CH_idx = sss[np.argmax(ppp)]
                            sss = sss.tolist()
                            sss.remove(max_priority_CH_idx)
                            # 加锁
                            for s in sss:
                                newC[i, s] = 0
                                newC[s, i] = 0

    print(f"\t[Done ({time.time()-sss_time:.3f})s] purning")



    # (*) LOG
    if results_path is not None and time_k is not None:
        # v. 求拉普拉斯矩阵
        sss_time = time.time()
        L = newC.copy()
        row, col = np.diag_indices_from(L)
        L[row, col] = -1. * np.sum(L, axis=1)
        L = -1 * L      # L再取反后，对角线是正数
        eigenvalue, featurevector = np.linalg.eig(L)
        second_smallest_eigenvalue_of_backbone_L = np.sort(eigenvalue)[1]
        log_each_step(results_path, time_k, newC, L)

        print(f"\t[Done ({time.time()-sss_time:.3f})s] 计算拉式矩阵")

    # vi. 计算候选点
    sss_time = time.time()

    CWPs = []
    for i in range(agent_num):
        # 每个智能体根据主干网络构建好后的1-hop邻居
        i_1_hop_nei_idx = _1hop_nei_idxs[i]
        nei_num = len(i_1_hop_nei_idx)

        # 汇总邻居的信息
        kkk = []
        for i in i_1_hop_nei_idx:
            kkk.append(points[i])
        kkk = np.array(kkk)

        # 计算候选点
        if len(kkk) == 0:  # 没邻居
            tmp_xy = points[i]
        else:  # 有邻居
            tmp_xy = (np.max(kkk, axis=0) + np.min(kkk, axis=0)) / 2

        # 每个智能体一个候选点坐标
        CWPs.append(tmp_xy)
    CWPs = np.array(CWPs)
    print(f"\t[Done ({time.time()-sss_time:.3f})s] 计算候选点")

    # vii. 计算约束集并求解最凸问题
    sss_time = time.time()

    for i in range(agent_num):
        # results = calc_CS(CWPs[i], points, i, newC, r_c, r_m)
        # next_points[i] = results.x
        next_points[i] = CWPs[i]
    print(f"\t[Done ({time.time()-sss_time:.3f})s] 移动")

    # viii. 保存过程文件
    sss_time = time.time()


    if results_path is not None and time_k is not None:
        types = np.array(types)
        np.save(f'./{results_path}/C_{time_k}.npy', C)
        np.save(f'{results_path}/L_{time_k}.npy', L)
        np.save(f'{results_path}/total_types_{time_k}.npy', types)
        np.save(f'./{results_path}/total_locations_{time_k}.npy', points)
        with open(f'{results_path}/eigenvalue.txt', 'a') as file:
            file.write(str(time_k) + "," + str(second_smallest_eigenvalue_of_backbone_L) + "\n")

    if ax is not None:
        vis_HSBMAS(ax, points, types, newC)
        plt.savefig(os.path.join(results_path, f"{time_k}.png"), dpi=500)
        plt.savefig(os.path.join(results_path, f"{time_k}.svg"))



    print(f"\t[Done ({time.time()-sss_time:.3f})s] 可视化及保存相关过程文件")


    # 动力学演化
    return next_points, newC