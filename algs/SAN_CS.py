# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/16 9:45

import numpy as np
from algs.vis import *
from algs.convex_minimize import calc_CS
from algs.log_each_step import log_each_step
from algs.detect_nei import detect_nei

from constants import *

def evolve(points, r_c, r_m, results_path=None, k=None, ax=None):

    agent_num = len(points)
    h = 1 / agent_num       # gaining parameter

    # (i) it detects its neighbors according to G;
    A, L = detect_nei(agent_num, points, r_c)

    # (*) LOG
    if results_path is not None and k is not None:
        log_each_step(results_path, k, A, L)

    # (ii) evolve
    L = np.mat(L)
    points = np.mat(points)
    next_points = points - h * L * points
    next_points = np.array(next_points)
    points = np.array(points)

    # (iii)
    for i in range(agent_num):
        results = calc_CS(next_points[i], points, i, A, r_c, r_m)
        next_points[i] = results.x

    if ax is not None:
        points = np.array(points)
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)


    return next_points, A
