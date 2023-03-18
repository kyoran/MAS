import os
import numpy as np

from algs.union_find import *


def calc_cluster_num(A):
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

    cluster_num = np.unique(groups).shape[0]
    return cluster_num


def log_each_step(results_path, step, A, L):
    if L is not None:
        # format
        one_line_format = "{},{},{},{},{}\n"
        # np.save(f'{results_path}/L_{k}.npy', L)
        # 感知矩阵C对应的边数
        C_edge_num = A.sum() / 2
        # 簇数
        cluster_num = calc_cluster_num(A)
        # L矩阵所对应的边数
        L_edge_num = L[np.diag_indices_from(L)].sum() / 2
        # L矩阵第二小特征值
        eigenvalue, featurevector = np.linalg.eig(L)
        second_smallest_eigenvalue_of_backbone_L = np.sort(eigenvalue)[1]

        with open(os.path.join(results_path, "0log.txt"), 'a') as file:
            file.write(one_line_format.format(
                step, C_edge_num, L_edge_num, cluster_num, second_smallest_eigenvalue_of_backbone_L
            ))

    else:
        # format
        one_line_format = "{},{},{}\n"
        # 感知矩阵C对应的边数
        C_edge_num = A.sum() / 2
        cluster_num = calc_cluster_num(A)

        with open(os.path.join(results_path, "0log.txt"), 'a') as file:
            file.write(one_line_format.format(
                step, C_edge_num, cluster_num
            ))
