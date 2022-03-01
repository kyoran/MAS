# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 21:36

import os
import glob
import shutil
import importlib
import matplotlib.pyplot as plt

from algs.is_end import *


if __name__ == '__main__':

    # 初始化
    plt.rc('font', family='Times New Roman')
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    # plt.ion()
    points = np.array([

        [-1.8, 0.35],
        [-2.0, -0.8],

        [0.7, 0.666],
        [0.86, -0.58],
        [1.2, -0.05],

        # GW
        [-0.333, 0],

        # DW
        [-0.8, 0.5],
        [-0.85, -0.6],

        # CH
        [-1.4, 0],
        [0.4, 0],

    ])

    end_threshold = 1e-2
    algorithm_type = "HSBMAS"
    results_path = rf"./results/diagram/{algorithm_type}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    else:
        shutil.rmtree(results_path)
        os.mkdir(results_path)
    print("\tlog save path:", results_path)
    alg = importlib.import_module(rf'algs.{algorithm_type}')

    k = 0

    while True:
        k += 1

        ax.cla()
        next_points, last_A = alg.evolve(points, 1.2, 0.04, results_path, k, ax)

        if is_end(last_A, k, points, end_threshold):
            break
        else:
            points = next_points
            # plt.pause(0.01)