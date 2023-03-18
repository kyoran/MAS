
import numpy as np


from algs.convex_minimize import calc_CS
from algs.vis import *
from algs.detect_nei import detect_nei
from algs.motifx.motifx import MotifX
from algs.log_each_step import log_each_step

from constants import *


def evolve(points, r_c, r_m, results_path=None, k=None, ax=None):

    next_points = points.copy()
    agent_num = len(points)
    h = 1 / agent_num       # gaining parameter

    # (i) it detects its neighbors according to G;
    alpha = 0.8
    A, _ = detect_nei(agent_num, points, r_c)
    M = MotifX(A).M4().toarray().astype(np.float32)
    W = (1-alpha) * A + alpha * M

    #
    W_r = np.reciprocal(W)
    W_r[W_r == np.inf] = 0
    W_r = W_r.astype(np.float32)
    # print("W_r:", W_r)
    #
    # D = np.diag(np.sum(W, axis=0)).astype(np.float32)
    D = np.diag(np.sum(A, axis=0)).astype(np.float32)
    #
    D_r = np.diag(np.sum(W_r, axis=0)).astype(np.float32)
    D_r_head_1 = np.reciprocal(D_r)
    D_r_head_1[D_r_head_1 == np.inf] = 0

    # print(D_r_head_1)
    # D_r = np.reciprocal(D)
    # D_r[D_r == np.inf] = 0
    # D_r = D_r.astype(np.float32)
    D = np.mat(D)
    D_r_head_1 = np.mat(D_r_head_1)
    D_r = np.mat(D_r)
    W_r = np.mat(W_r)
    # L = D - D * np.linalg.inv(D_r) * W_r
    L = D - D * D_r_head_1 * W_r

    # (*) LOG
    if results_path is not None and k is not None:
        log_each_step(results_path, k, A, L)

    # (ii)
    # L = np.mat(D) - np.mat(D / D_r) * np.mat(W_r)
    L = np.mat(L)
    points = np.mat(points)
    next_points = (np.mat(np.identity(agent_num)) - h * L) * points
    next_points = np.array(next_points)
    points = np.array(points)

    # (iv) 连通性约束集，如果需要就接触注释
    # for i in range(agent_num):
    #     results = calc_CS(next_points[i], points, i, A, r_c, r_m)
    #     next_points[i] = results.x

    if ax is not None:
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)

    # 动力学演化
    return next_points, A
