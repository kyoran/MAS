# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 21:40

import numpy as np

def is_end(A, k, total_locations, threshold=1e-2):
    agent_num = len(total_locations)
    flags = []
    for i in range(agent_num):
        nei_idx = np.where(A[i]==1)[0].tolist()  # 相连的邻居
        nei_idx.append(i)
        nei_idx = np.array(nei_idx)
        nei_locations = total_locations[nei_idx]    # 包含自己的

        mean_location = nei_locations.mean(axis=0)

        # 所有智能体到平均位置的距离
        total_dis = np.linalg.norm(nei_locations - mean_location, axis=1)
        mean_dis = np.mean(total_dis)

        if mean_dis < threshold:
            flags.append(True)
        else:
            flags.append(False)


    flags = np.array(flags)
    false_idx = np.where(flags == False)[0]
    print("\rat time:", k, "false_num:", len(false_idx), end="")
    return flags.all()