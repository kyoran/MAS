# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/7/19 22:14

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# init_ax和plot_ax是用于除了HSBMAS以外的算法的

def init_ax(ax):
    ax.axis('on')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.tick_params(
        # axis='both',
        # which='both',
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
    )
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

def plot_ax(ax, points, A, node_size=100):
    ax.scatter(points[:, 0], points[:, 1])
    G = nx.from_numpy_matrix(A)
    nx.draw(G, pos=points, ax=ax, edge_color="#1f77b4", node_color="#1f77b4", node_size=node_size)


# 画legend
TYPE2COLOR = {"CH": "orangered", "DW": "seagreen", "GW": "slateblue", "NB": "darkslategray"}
TYPE2MARKER = {"CH": "s", "DW": "d", "GW": "*", "NB": "o"}
TYPE2ALPHA = {"CH": 1, "DW": 1, "GW": 1, "NB": 0.5}

abbr_type = {r"CH (ClusterHead)": "CH",
             r"DW (DoorWay)": "DW",
             r"GW (GateWay)": "GW",
             r"NB (Non-Backbone)": "NB"}

def vis_HSBMAS(ax, points, types, A):
    agent_num = len(A)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_aspect(1. / ax.get_data_ratio())

    for one_legend in abbr_type.keys():
        ax.scatter(
            -100, -100,
            c=TYPE2COLOR[abbr_type[one_legend]],
            marker=TYPE2MARKER[abbr_type[one_legend]],
            label=one_legend
        )

    # 画所有智能体点
    CH_IDX = types == "CH"
    DW_IDX = types == "DW"
    GW_IDX = types == "GW"
    NB_IDX = types == "NB"
    ax.scatter(points[CH_IDX, 0], points[CH_IDX, 1], s=8,
               marker=TYPE2MARKER["CH"], color=TYPE2COLOR["CH"], alpha=TYPE2ALPHA["CH"], zorder=10)
    ax.scatter(points[DW_IDX, 0], points[DW_IDX, 1], s=8,
               marker=TYPE2MARKER["DW"], color=TYPE2COLOR["DW"], alpha=TYPE2ALPHA["DW"], zorder=9)
    ax.scatter(points[GW_IDX, 0], points[GW_IDX, 1], s=8,
               marker=TYPE2MARKER["GW"], color=TYPE2COLOR["GW"], alpha=TYPE2ALPHA["GW"], zorder=8)
    ax.scatter(points[NB_IDX, 0], points[NB_IDX, 1], s=8,
               marker=TYPE2MARKER["NB"], color=TYPE2COLOR["NB"], alpha=TYPE2ALPHA["NB"], zorder=7)
    # 连线
    color = "darkturquoise"
    for i in range(agent_num):
        nei_idx = np.where(A[i, :] == 1)[0]
        for one_nei_idx in nei_idx:
            ax.plot(
                [points[i, 0], points[one_nei_idx, 0]],
                [points[i, 1], points[one_nei_idx, 1]],
                color=color, alpha=0.8, zorder=-1, linewidth=0.6
            )
    leg = ax.legend(loc='upper right')
    leg.set_zorder(-2)
