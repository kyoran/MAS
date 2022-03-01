# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 10:06

"""
Circumcenter Algorithm: Each agent performs:

(i) it detects its neighbors according to G;
(ii) it computes the circumcenter of the point set comprised of its neighbors and of itself;
(iii) it moves toward this circumcenter while maintaining connectivity with its neighbors.

"""


from algs.smallestenclosingcircle import make_circle
from algs.convex_minimize import calc_CS
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
        results = calc_CS(np.array([c[0], c[1]]), points, i, A, r_c, r_m)
        # print(results.x)
        next_points[i] = results.x

    L = A.copy()
    row, col = np.diag_indices_from(L)
    L[row, col] = -1. * np.sum(L, axis=1)
    L = -1 * L  # L再取反后，对角线是正数
    eigenvalue, featurevector = np.linalg.eig(L)
    second_smallest_eigenvalue_of_backbone_L = np.sort(eigenvalue)[1]
    if results_path is not None and k is not None:
        np.save(f'{results_path}/L_{k}.npy', L)
        with open(f'{results_path}/eigenvalue.txt', 'a') as file:
            file.write(str(k) + "," + str(second_smallest_eigenvalue_of_backbone_L) + "\n")


    if ax is not None:
        plot_ax(ax, points, A, node_size=node_size)
        init_ax(ax)
        plt.savefig(os.path.join(results_path, f"{k}.png"), dpi=500)

    # 动力学演化
    return next_points, A


if __name__ == '__main__':

    from matplotlib.patches import Circle
    from scipy.spatial import ConvexHull

    points = np.random.rand(10, 2)  # 30 random points in 2-D
    hull = ConvexHull(points)
    import matplotlib.pyplot as plt

    plt.plot(points[:, 0], points[:, 1], 'o')

    print(hull.simplices)
    print(type(hull.simplices))

    c = make_circle(points)
    print(c)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(c[0], c[1], color="red", s=55)
    cir1 = Circle(xy = (c[0], c[1]), radius=c[2], alpha=0.5)
    ax.add_patch(cir1)

    for simplex in hull.simplices:
       ax.plot(points[simplex,0], points[simplex,1], 'k-')
    plt.show()
