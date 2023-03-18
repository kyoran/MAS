# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/16 9:45

import numpy as np
from algs.vis import *
from constants import *
from algs.log_each_step import log_each_step

def evolve(points, r_c, r_m, results_path=None, k=None, ax=None):

    agent_num = len(points)
    h = 1/agent_num

    # (i) it detects its neighbors according to G;
    A = np.zeros(shape=(agent_num, agent_num), dtype=np.int32)
    for i in range(agent_num):
        for j in range(i+1, agent_num):
            if np.linalg.norm(points[i]-points[j]) <= r_c:
                A[i, j] = 1
                A[j, i] = 1

    L = A.copy()
    row, col = np.diag_indices_from(L)
    L[row, col] = -1. * np.sum(L, axis=1)
    L = -1 * L  # L再取反后，对角线是正数

    if results_path is not None and k is not None:
        log_each_step(results_path, k, A, L)


    L = np.mat(L)
    points = np.mat(points)

    # (ii)
    next_points = points - h * L * points
    next_points = np.array(next_points)

    if ax is not None:
        points = np.array(points)
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)


    return next_points, A
