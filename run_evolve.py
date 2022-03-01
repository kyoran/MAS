# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 21:36

import importlib
import matplotlib.pyplot as plt

from algs.is_end import *
from constants import *


if __name__ == '__main__':

    # 初始化
    plt.rc('font', family='Times New Roman')
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    # plt.ion()
    points = np.load(dataset_path)

    alg = importlib.import_module(rf'algs.{algorithm_type}')
    # from algs.CA import *

    k = 0

    while True:
        k += 1

        ax.cla()
        next_points, last_A = alg.evolve(points, 0.5, 0.5, results_path, k, ax)

        if is_end(last_A, k, points, end_threshold):
            break
        else:
            points = next_points
            # plt.pause(0.01)