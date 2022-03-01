# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/2/12 14:40

import os
import sys
sys.path.append(os.getcwd())

import glob
import numpy as np
import collections
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter, LogFormatter, FixedLocator
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

from init_fig import *

plt.rc('font', family='Times New Roman')

"""需要修改的路径"""
log_idx = "010"
log_paths = [
    f"./Agents-200/{log_idx}*/HSBMAS/",
    f"./Agents-400/{log_idx}*/HSBMAS/",
    f"./Agents-600/{log_idx}*/HSBMAS/",
    f"./Agents-800/{log_idx}*/HSBMAS/",
    f"./Agents-1000/{log_idx}*/HSBMAS/",
]
labels = [
    rf"T{log_idx.lstrip('0')} ($\rho=2$)",
    rf"T{log_idx.lstrip('0')} ($\rho=4$)",
    rf"T{log_idx.lstrip('0')} ($\rho=6$)",
    rf"T{log_idx.lstrip('0')} ($\rho=8$)",
    rf"T{log_idx.lstrip('0')} ($\rho=10$)",
]

total_bbs = []
max_step = 0

for idx, one_log_path in enumerate(log_paths):
    print("at:", one_log_path)
    # C_filename_lst = glob.glob(f"{one_log_path}/C_*")
    L_filename_lst = glob.glob(f"{one_log_path}/L_*")
    # C_filename_lst.sort(key=lambda x: int(x[x.rindex("_")+1: x.rindex(".")]))
    L_filename_lst.sort(key=lambda x: int(x[x.rindex("_")+1: x.rindex(".")]))

    # min_len = min(len(C_filename_lst), len(L_filename_lst))
    # C_filename_lst = C_filename_lst[:min_len]

    steps = list(range(1, len(L_filename_lst)+1))
    # TYPE2COLOR = {"CH": "orangered", "DW": "seagreen", "GW": "slateblue", "NB": "darkslategray"}

    # _1_hop_connect = []     # 原始感知半径内的连接
    _bb_connect = []        # 主干网络连接
    # last_L = None
    for i in range(len(L_filename_lst)):
        # one_C = np.load(C_filename_lst[i])
        one_L = np.load(L_filename_lst[i])
        # last_L = one_L
        #
        # _1_hop_connect.append(len(np.where(one_C == 1)[0])/2)
        _bb_connect.append(np.sum(np.diagonal(one_L)).astype(np.int32)/2)
    # print(_1_hop_connect)
    # print(_bb_connect)
    # print("last_1_hop:", _1_hop_connect[-1])
    print("last_bb:", _bb_connect[-1])
    total_bbs.append(_bb_connect)
    if len(_bb_connect) > max_step:
        max_step = len(_bb_connect)
    # print(np.diagonal(last_L))


print("max_len:", max_step)

# 补充数据
for idx in range(5):
    total_bbs[idx] += [total_bbs[idx][-1]] * (max_step - len(total_bbs[idx]))

ax = plt.subplot(111)

for idx in range(5):
    # plt.plot(steps, _1_hop_connect, color=colors[idx], label="1-hop connections")
    ax.plot(list(range(max_step)), total_bbs[idx], color=colors[idx], label=labels[idx])

for idx in range(5):
    # ax.plot(max_step-1, total_bbs[idx][-1], ".r")
    offsetbox = TextArea(total_bbs[idx][-1])
    # ab = AnnotationBbox(offsetbox, [max_step-1, total_bbs[idx][-1]],
    #                     xybox=(max_step, total_bbs[idx][-1]),
    #                     xycoords='data',
    #                     boxcoords=("axes fraction", "data"),
    #                     box_alignment=(0., 0.5),
    #                     arrowprops=dict(arrowstyle="->"))
    ab = AnnotationBbox(offsetbox, [max_step-1, total_bbs[idx][-1]],
                        xybox=(45, 20),
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"),
                        fontsize=5,
                        )
    ax.add_artist(ab)
    ax.add_artist(ab)

# 样式
ax.set_xlabel("Evolution times $k$")
ax.set_ylabel("The number of backbone-based connections")
# plt.yscale('log')
# plt.yscale('symlog')
# plt.yscale('logit')
# plt.gca().yaxis.set_major_locator(FixedLocator(np.arange(0, 1e5, 1e4)))
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.gca().yaxis.set_major_locator(MultipleLocator(3e3))
# plt.gca().yaxis.set_minor_formatter(LogFormatter())
ax.legend(loc='best')
# plt.grid(True)
# plt.title("Original 1-hop connections V.S. Backbone-based connections")
plt.savefig(f"./vis/connect_num/density_bb_cons_{log_idx}.png", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/connect_num/density_bb_cons_{log_idx}.pdf", bbox_inches='tight', pad_inches=0)
plt.show()

