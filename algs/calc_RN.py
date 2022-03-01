# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/10 22:15

import numpy as np

def calc_RN(A, locations):

    n = len(locations)

    # RN的意思就是任意i与j组成的B中，没有其他的智能体
    RN = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(i+1, n):
            # if i != j:
            i_j_dis = np.linalg.norm(locations[i] - locations[j])
            # if i_j_dis <= r_c:
            #     flag = False
            #     break
            flag = True
            for k in range(n):
                if k != i and k != j:
                    k_j_dis = np.linalg.norm(locations[k] - locations[i])
                    k_i_dis = np.linalg.norm(locations[k] - locations[j])

                    # if k_i_dis <= i_j_dis and k_j_dis <= i_j_dis:
                    if k_i_dis < i_j_dis and k_j_dis < i_j_dis:
                        flag = False
                        break
            if flag:
                RN[i, j] = 1
                RN[j, i] = 1
    A = np.array(A, dtype=np.int32)
    RN = np.array(RN, dtype=np.int32)
    RN = RN & A
    return RN