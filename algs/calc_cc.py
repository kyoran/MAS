# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/11 15:58

import numpy as np

def find(x, pre):
    r = x
    while pre[r] != r:
        r = pre[r]  # 找到前导节点
    i = x
    while i != r:
        j = pre[i]
        pre[i] = r
        i = j
    return r

def join(x, y, pre):
    a = find(x, pre)
    b = find(y, pre)
    if a != b:
        pre[a] = b


def calc_cc(C):
    edge_lst = []
    node_lst = list(range(len(C)))
    # C = np.load(f"{log_path}/C_{step}.npy", encoding='bytes', allow_pickle=True)
    for i in node_lst:
        for j in node_lst:  # 结尾点
            # if i != j and j in agent_objs[i].workfor:
            if i != j and C[j, i] == 1:
                edge_lst.append([i, j])

    # 4. 初始化前导节点 pre 数据结构
    pre = [i for i in range(len(node_lst))]
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

    cluster_num = np.unique(groups).shape
    return cluster_num