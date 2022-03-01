# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 10:07

from algs.smallestenclosingcircle import make_circle
from algs.convex_minimize import calc_CS
from algs.calc_RN import calc_RN
from algs.vis import *
from constants import *

import numpy as np
import matplotlib.pyplot as plt

def evolve(points, r_c, r_m, results_path=None, k=None, ax=None):

    next_points = points.copy()
    agent_num = len(points)

    # (i) it detects its neighbors according to G;
    A = np.zeros(shape=(agent_num, agent_num), dtype=np.int32)
    for i in range(agent_num):
        for j in range(i+1, agent_num):
            if np.linalg.norm(points[i]-points[j]) <= r_c:
                A[i, j] = 1
                A[j, i] = 1

    RN_A = calc_RN(A, points)

    # (ii) it computes the circumcenter of the point set comprised of its neighbors and of itself;
    # circumcenters = []
    for i in range(agent_num):
        nei_idx = np.where(A[i]==1)[0].tolist()
        if nei_idx == 0:    # 没有邻居
            # circumcenters.append(points[i])
            continue

        nei_idx.append(i)
        nei_and_i_idx = np.array(nei_idx)

        # circumcenter
        c = make_circle(points[nei_and_i_idx])
        # circumcenters.append([c[0], c[1]])

        # (iii) it moves toward this circumcenter while maintaining connectivity with its neighbors.
        # https://www.jb51.net/article/180064.htm
        # http://www.cocoachina.com/articles/90935
        # 每个智能体求约束集，计算CA的最优点
        results = calc_CS(np.array([c[0], c[1]]), points, i, RN_A, r_c, r_m)
        # print(results.x)
        next_points[i] = results.x

    if ax is not None:
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)

    # 动力学演化
    return next_points, A
