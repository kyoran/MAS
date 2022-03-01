# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 13:36

import numpy as np
from scipy.optimize import minimize


# np.random.seed(6)

def calc_CS(circumcenter, points, i, A, r_c, r_m):

    nei_idx = np.where(A[i]==1)[0]
    def obj_fun(args):
        # 求i到外心的向量 在约束集上的最远点
        xi, yi, xc, yc = args   # i是智能体i的坐标，c是外心的坐标
        # 先用点向式表达线，然后y换成x，再表达出点到i坐标的距离
        # x = x
        # y = (yc-yi)/(xc-xi) * (x-xi) + yi
        # (x, ((yc-yi)/(xc-xi) * (x-xi) + yi)) - (xi, yi)
        # v = lambda x: - (((yc-yi)/(xc-xi) * (x-xi) + yi) - yi) ** 2 - (x-xi) ** 2
        v = lambda loc: - (loc[0]-xi) ** 2 - (loc[1]-yi) ** 2
        return v

    # 其中constr_fun是可调用的函数,使得 constr_fun >= 0
    args = (points[i, 0], points[i, 1], circumcenter[0], circumcenter[1])
    xi, yi, xc, yc = args
    constraints = []

    for j in nei_idx:
        # cfunc = lambda j: lambda loc=j: r/2 - np.linalg.norm(np.array(loc) - np.array([(xi+points[j,0])/2, (yi+points[j,1])/2]))
        # 每个邻居的约束
        cfunc = lambda j: lambda loc=j: r_c*r_c/4 - (loc[0]-(xi+points[j,0])/2)**2 - (loc[1]-(yi+points[j,1])/2)**2
        cdict = {
            'type': 'ineq',
            'fun': cfunc(j)
        }
        # 每个邻居的约束
        constraints.append(cdict)
    #
    # constraints.append(
    #     # 约束1：loc在i的r_m圆心内
    #     # {'type': 'ineq', 'fun': lambda loc: -(loc[0]-xi)**2 - (loc[1]-yi)**2 + r**2},
    # )
    th = 1e-6
    constraints.append(
        # 约束2：loc在xi, yi, xc, yc线上 （点向式）
        {'type': 'eq', 'fun': lambda loc: loc[1] - yi - (yc - yi)/(xc - xi + th) * (loc[0] - xi + th)},
    )
    constraints.append(
        # 约束3：xi->loc 和 xi->xc同向 a*b>=0
        {'type': 'ineq', 'fun': lambda loc: (loc[0]-xi)*(xc-xi) + (loc[1]-yi)*(yc-yi)},
    )
    constraints.append(
        # 约束4：xi<->xc loc[0]在他们之间
        {'type': 'ineq', 'fun': lambda loc: (loc[0]-xi)*(xc-loc[0])},
    )
    constraints.append(
        # 约束4：yi<->yc loc[1]在他们之间
        {'type': 'ineq', 'fun': lambda loc: (loc[1]-yi)*(yc-loc[1])},
    )
    constraints.append(
        # 约束5：loc在以i为中心，r_m为半径的圆内
        {'type': 'ineq', 'fun': lambda loc: r_m - np.linalg.norm((loc[0]-xi)**2 + (loc[1]-yi)**2)},
    )

    result = minimize(
        fun=obj_fun(args),
        x0=np.array([xc, yc]),    # 起始点
        bounds=((xi-r_c, xi+r_c), (yi-r_c, yi+r_c)),
        method="SLSQP",
        options={
            # "disp": True,
            "maxiter": 20,
        },
        constraints=constraints,        # constraints默认是≥0的
    )
    return result
    # print(result)
    # print(result.success)
    # print("解为：", result.x)
    # print("外心：", c[0], c[1])

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    from smallestenclosingcircle import make_circle

    r = 3
    agent_num = 5

    L = 3
    points = np.random.uniform(-L, L, (agent_num-1, 2)).tolist()    # 30 random points in 2-D
    points.append([0, 0])
    points = np.array(points)

    c = make_circle(points)
    print(c)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect(1./ax.get_data_ratio())

    ax.scatter(points[:,0], points[:,1])
    ax.scatter(points[agent_num-1, 0], points[agent_num-1, 1], color="k", label="i's location")

    i_r = Circle(xy=(points[agent_num-1, 0], points[agent_num-1, 1]), radius=r, alpha=0.5)
    ax.add_patch(i_r)

    # A
    A = np.zeros(shape=(agent_num, agent_num), dtype=np.int32)
    for i in range(agent_num):
        for j in range(i+1, agent_num):
            if np.linalg.norm(points[i]-points[j]) <= r:
                A[i, j] = 1
                A[j, i] = 1
    print(points.shape)
    # c
    i = agent_num-1
    nei_idx = np.where(A[i]==1)[0].tolist()
    nei_and_i_idx = nei_idx.copy()
    nei_and_i_idx.append(i)
    nei_and_i_idx = np.array(nei_and_i_idx)
    print("nei_idx:", nei_idx)
    print("nei_and_i_idx:", nei_and_i_idx)

    # circumcenter
    c = make_circle(points[nei_and_i_idx])
    # c = np.array([-2,-2, 1])
    ax.scatter(c[0], c[1], color="red", label="circumcenter")
    i_nei_circumcenter = Circle(xy = (c[0], c[1]), radius=c[2], color='red', alpha=0.3)
    ax.add_patch(i_nei_circumcenter)

    # nei r/2 circle
    for one_nei_idx in nei_idx:
        i_nei_circle = Circle(xy=((points[one_nei_idx, 0]+points[i, 0])/2, (points[one_nei_idx, 1]+points[i, 1])/2), radius=r/2, color='green', alpha=0.3, zorder=-111)
        ax.add_patch(i_nei_circle)


    # 【开始优化】

    # 先判断有没有邻居

    # 没有就不优化了
    # 最小化函数
    def fun(args):
        # 求i到外心的向量 在约束集上的最远点
        xi, yi, xc, yc = args   # i是智能体i的坐标，c是外心的坐标
        # 先用点向式表达线，然后y换成x，再表达出点到i坐标的距离
        # x = x
        # y = (yc-yi)/(xc-xi) * (x-xi) + yi
        # (x, ((yc-yi)/(xc-xi) * (x-xi) + yi)) - (xi, yi)
        # v = lambda x: - (((yc-yi)/(xc-xi) * (x-xi) + yi) - yi) ** 2 - (x-xi) ** 2
        v = lambda loc: - (loc[0]-xi) ** 2 - (loc[1]-yi) ** 2
        return v

    # 其中constr_fun是可调用的函数,使得 constr_fun >= 0
    args = (points[i, 0], points[i, 1], c[0], c[1])
    xi, yi, xc, yc = args
    constraints = []
    aaa = 0

    for j in nei_idx:
        # cfunc = lambda j: lambda loc=j: r/2 - np.linalg.norm(np.array(loc) - np.array([(xi+points[j,0])/2, (yi+points[j,1])/2]))
        cfunc = lambda j: lambda loc=j: r*r/4 - (loc[0]-(xi+points[j,0])/2)**2 - (loc[1]-(yi+points[j,1])/2)**2
        cdict = {
            'type': 'ineq',
            # 'fun': lambda loc: -(loc[0]-(xi+points[j,0])/2)**2 - (loc[1]-(yi+points[j,1])/2)**2 + (r/2)**2
            # 'fun': lambda loc: r*r/4 - (loc[0]-(xi+points[j,0])/2)**2 - (loc[1]-(yi+points[j,1])/2)**2
            # 'fun': lambda loc: r*r/4 - (loc[0]-(xi+points[nei_idx[j_idx],0])/2)**2 - (loc[1]-(yi+points[nei_idx[j_idx],1])/2)**2
            # 'fun': lambda i: lambda loc: r/2 - np.linalg.norm(np.array(loc) - np.array([(xi+points[j,0])/2, (yi+points[j,1])/2]))
            'fun': cfunc(j)
            # 'fun': cfun
        }
        # 每个邻居的约束
        constraints.append(cdict)
        # print("j=", j, points[j], np.array([(xi+points[j,0])/2, (yi+points[j,1])/2]))
        # aaa += 1
        # break
        # if aaa == 2:
        #     break

    # constraints.append(
    #     # 约束1：loc在i的r_m圆心内
    #     # {'type': 'ineq', 'fun': lambda loc: -(loc[0]-xi)**2 - (loc[1]-yi)**2 + r**2},
    # )
    th = 1e-6
    constraints.append(
        # 约束2：loc在xi, yi, xc, yc线上 （点向式）
        # {'type': 'eq', 'fun': lambda loc: (loc[0] - xi) / (xc - xi) - (loc[1] - yi) / (yc - yi)},
        {'type': 'eq', 'fun': lambda loc: loc[1] - yi - (yc - yi)/(xc - xi+th) * (loc[0] - xi+th)},
    )
    th = 1e-3
    constraints.append(
        # 约束3：xi->loc 和 xi->xc同向 a*b>=0
        {'type': 'ineq', 'fun': lambda loc: (loc[0]-xi)*(xc-xi) + (loc[1]-yi)*(yc-yi)},
    )
    constraints.append(
        # 约束4：xi<->xc loc[0]在他们之间
        {'type': 'ineq', 'fun': lambda loc: (loc[0] - xi) * (xc - loc[0])},
    )
    constraints.append(
        # 约束4：yi<->yc loc[1]在他们之间
        {'type': 'ineq', 'fun': lambda loc: (loc[1] - yi) * (yc - loc[1])},
    )
    print(len(constraints))

    k = (yc-yi) / (xc-xi)
    result = minimize(
        fun=fun(args),
        x0=np.array([xc, yc]),    # 起始点
        # x0=np.array([xi, yi]),
        # x0=np.array([k*0.001+xi, k*0.001+yi]),
        bounds=((xi-r, xi+r), (yi-r, yi+r)),
        # method="L-BFGS-B",
        # method="BFGS",
        method="SLSQP",
        # method="COBYLA",
        # tol=1e-3,
        options={
            # "disp": True,
            "maxiter": 20,
        },
        constraints=constraints,
    )

    print(result)
    print(result.success)
    print("解为：", result.x)
    print("外心：", c[0], c[1])

    ax.scatter(result.x[0], result.x[1], color='yellow', s=3, label="result", zorder=111)
    # print(result.fun)
    plt.legend()
    plt.show()
