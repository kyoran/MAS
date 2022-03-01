# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 10:07

from algs.smallestenclosingcircle import make_circle
from algs.convex_minimize import calc_CS
from algs.vis import *
from constants import *

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

def evolve(points, r_c, r_m, results_path=None, k=None, ax=None):

    next_points = points.copy()
    agent_num = len(points)

    # (i) 根据感知半径获取所有邻居
    A = np.zeros(shape=(agent_num, agent_num), dtype=np.int32)
    for i in range(agent_num):
        for j in range(i+1, agent_num):
            if np.linalg.norm(points[i]-points[j]) <= r_c:
                A[i, j] = 1
                A[j, i] = 1

    # (ii) 根据自身和邻居构建凸包
    hull_A = np.zeros(shape=(agent_num, agent_num), dtype=np.int32)
    for i in range(agent_num):
        nei_idx = np.where(A[i]==1)[0].tolist()

        if nei_idx == 0:    # 没有邻居
            # circumcenters.append(points[i])
            continue

        nei_idx.append(i)
        nei_and_i_idx = np.array(nei_idx)
        nei_and_i_pos = points[nei_and_i_idx]


        # 【TP2】circumcenter
        c = make_circle(points[nei_and_i_idx])

        try:
            # 求解邻居和自身状态集合的凸包
            hull = ConvexHull(nei_and_i_pos)
            # 凸包的顶点下标集合
            hullver = hull.vertices.tolist()
            # 凸包的顶点坐标集合
            hullpos = nei_and_i_pos[hullver]

            # 【TP1】
            # 0 -d01- 1 -d12- 2 -
            # |                 |
            # ------d20----------
            hullpos_len = len(hullpos)
            ds = []
            for one_hullpos_idx, one_hullpos in enumerate(hullpos):
                tmp_d = np.linalg.norm(hullpos[(one_hullpos_idx + 1) % hullpos_len] - one_hullpos)
                ds.append(tmp_d)
            # print("ds:", ds)

            total_w = np.sum(ds) * 2  # 求CHVSM权重的分母
            ws = []
            for w_idx in range(hullpos_len):
                tmp_w = (ds[w_idx] + ds[(w_idx + 1) % hullpos_len]) / total_w
                ws.append(tmp_w)
            # print("ws:", ws)
            # print("sum of ws:", np.sum(ws))   # 和为1
            first_w = ws.pop(-1)
            ws.insert(0, first_w)  # 这样w就跟邻居的下标都对应上了
            # print("ws:", ws)
            TP1 = np.array([0., 0.])
            for j in range(len(hullpos)):
                TP1 += ws[j] * hullpos[j]

            # (iii) it moves toward this circumcenter while maintaining connectivity with its neighbors.
            # https://www.jb51.net/article/180064.htm
            # http://www.cocoachina.com/articles/90935
            # 每个智能体求约束集，计算CA的最优点
            for j in nei_and_i_idx[hullver]:
                if j != i:
                    hull_A[i, j] = 1
                    hull_A[j, i] = 1
            # results_tp1 = calc_CS(TP1, points, i, hull_A, r_c, r_m)
            results_tp1 = calc_CS(TP1, points, i, A, r_c, r_m)
            results_tp2 = calc_CS(np.array([c[0], c[1]]), points, i, A, r_c, r_m)

            dis_tp1_i = np.linalg.norm(results_tp1.x-points[i])
            dis_tp2_i = np.linalg.norm(results_tp2.x-points[i])
            if dis_tp1_i > dis_tp2_i:
                next_points[i] = results_tp1.x
            else:
                next_points[i] = results_tp2.x

        except scipy.spatial.qhull.QhullError:
            # 只有一个邻居和自身，构建不了凸包，所以直接用外心算法
            results = calc_CS(np.array([c[0], c[1]]), points, i, A, r_c, r_m)
            next_points[i] = results.x

    if ax is not None:
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)

    # 动力学演化
    return next_points, A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy.spatial import ConvexHull


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_aspect(1. / ax.get_data_ratio())

    agent_num = 6
    points = np.random.rand(agent_num-1, 2)  # 30 random points in 2-D
    points = points.tolist()
    points.append([0.5, 0.5])
    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], 'o')


    i_idx = len(points) - 1     # 最后一个点是i

    r = 0.5
    A = np.zeros(shape=(agent_num, agent_num), dtype=np.int32)
    for i in range(agent_num):
        for j in range(i + 1, agent_num):
            if np.linalg.norm(points[i] - points[j]) <= r:
                A[i, j] = 1
                A[j, i] = 1

    nei_idx = np.where(A[i_idx] == 1)[0].tolist()
    nei_idx.append(i_idx)
    nei_and_i_idx = np.array(nei_idx)
    nei_and_i_pos = points[nei_and_i_idx]


    # 【下面开始求CHVSM】
    # 求解邻居和自身状态集合的凸包
    hull = ConvexHull(nei_and_i_pos)
    # 凸包的顶点下标集合
    hullver = hull.vertices.tolist()
    print("i的凸包的顶点集:", nei_and_i_idx[hullver])
    # 凸包的顶点坐标集合
    hullpos = nei_and_i_pos[hullver]
    ax.plot(hullpos[:, 0], hullpos[:, 1], 'r--', lw=2)

    # 0 -d01- 1 -d12- 2 -
    # |                 |
    # ------d20----------
    hullpos_len = len(hullpos)
    ds = []
    for one_hullpos_idx, one_hullpos in enumerate(hullpos):
        tmp_d = np.linalg.norm(hullpos[(one_hullpos_idx+1) % hullpos_len] - one_hullpos)
        ds.append(tmp_d)
    print("ds:", ds)

    total_w = np.sum(ds) * 2    # 求CHVSM权重的分母
    ws = []
    for w_idx in range(hullpos_len):
        tmp_w = (ds[w_idx] + ds[(w_idx+1) % hullpos_len]) / total_w
        ws.append(tmp_w)
    print("ws:", ws)
    print("sum of ws:", np.sum(ws))
    first_w = ws.pop(-1)
    ws.insert(0, first_w)  # 这样w就跟邻居的下标都对应上了
    print("ws:", ws)

    #
    TP1 = np.array([0., 0.])
    for j in range(len(hullpos)):
        TP1 += ws[j] * hullpos[j]
    print("TP1:", TP1)
    ax.scatter(TP1[0], TP1[1], color='white', s=50, label="CHVSM", zorder=111)

    for i in range(hullpos_len):
        ax.text(hullpos[i, 0], hullpos[i, 1], f'*')

    hull_A = np.zeros(shape=(agent_num, agent_num), dtype=np.int32)

    for j in nei_and_i_idx[hullver]:
        if j != i_idx:
            hull_A[i_idx, j] = 1
            hull_A[j, i_idx] = 1
        else:
            print("111111")
    print(hull_A[i_idx])
    print(hull_A)
    nei_and_i_idx = nei_and_i_idx.copy().tolist()
    nei_and_i_idx.remove(i_idx)
    for one_nei_idx in nei_and_i_idx:
        i_nei_circle = Circle(xy=((points[one_nei_idx, 0]+points[i_idx, 0])/2, (points[one_nei_idx, 1]+points[i_idx, 1])/2), radius=r/2, color='green', alpha=0.3, zorder=-111)
        ax.add_patch(i_nei_circle)
    results_tp1 = calc_CS(TP1, points, i_idx, hull_A, r, r)
    print("results_tp1:", results_tp1.x)
    ax.scatter(results_tp1.x[0], results_tp1.x[1], color='white', edgecolor='k', s=50, label="CHVSM_target", zorder=111)

    # 【圆心算法】
    c = make_circle(nei_and_i_pos)
    cir1 = Circle(xy=(c[0], c[1]), radius=c[2], alpha=0.5)
    ax.add_patch(cir1)
    print("Circumcenter:", c)
    ax.scatter(c[0], c[1], color='yellow', s=50, label="Circumcenter", zorder=111)

    results_c = calc_CS(np.array([c[0], c[1]]), points, i_idx, A, r, r)
    print("results_c:", results_c.x)
    ax.scatter(results_c.x[0], results_c.x[1], color='yellow', edgecolor='k', s=50, label="Circumcenter_target", zorder=111)
    # for one_nei_idx in nei_idx:
    #     i_nei_circle = Circle(xy=((points[one_nei_idx, 0]+points[i_idx, 0])/2, (points[one_nei_idx, 1]+points[i_idx, 1])/2), radius=r/2, color='green', alpha=0.3, zorder=-111)
    #     ax.add_patch(i_nei_circle)

    # for simplex in hull.simplices:
    #    ax.plot(points[simplex, 0], points[simplex,1], 'k-')
       # ax.text(points[simplex, 0], points[simplex,1], 'k-')
    ax.text(points[i_idx, 0], points[i_idx, 1], f'i={i_idx}')
    for each_point_idx in range(len(points)):
        ax.text(points[each_point_idx, 0], points[each_point_idx, 1], f'{each_point_idx}')

    plt.legend()
    pass
    plt.show()

