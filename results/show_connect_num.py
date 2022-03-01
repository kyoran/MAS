# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2020/11/17 20:08

import os
import glob
import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter, LogFormatter, FixedLocator

from init_fig import *

plt.rc('font', family='Times New Roman')

"""需要修改的路径"""
log_dir = "001log_uniform_600"
log_dir = "002log_ring_600"
log_dir = "003log_vase_600"
# log_dir = "004log_taiji_600"
# log_dir = "005log_circle_600"
# log_dir = "006log_triangle_600"
# log_dir = "007log_square_600"
# log_dir = "008log_arch_600"
# log_dir = "009log_neat_square_625"
# log_dir = "010log_neat_radiation_600"
log_path = f"./Agents-600/{log_dir}/HSBMAS"
C_filename_lst = glob.glob(f"{log_path}/C_*")
L_filename_lst = glob.glob(f"{log_path}/L_*")
C_filename_lst.sort(key=lambda x: int(x[x.rindex("_")+1: x.rindex(".")]))
L_filename_lst.sort(key=lambda x: int(x[x.rindex("_")+1: x.rindex(".")]))


min_len = min(len(C_filename_lst), len(L_filename_lst))
C_filename_lst = C_filename_lst[:min_len]
L_filename_lst = L_filename_lst[:min_len]

print("C_filename_lst:")
print(len(C_filename_lst), C_filename_lst)
print("L_filename_lst:")
print(len(L_filename_lst), L_filename_lst)


steps = list(range(1, len(L_filename_lst)+1))
TYPE2COLOR = {"CH": "orangered", "DW": "seagreen", "GW": "slateblue", "NB": "darkslategray"}


_1_hop_connect = []     # 原始感知半径内的连接
_bb_connect = []        # 主干网络连接
last_L = None
for i in range(len(L_filename_lst)):
    one_C = np.load(C_filename_lst[i])
    one_L = np.load(L_filename_lst[i])
    last_L = one_L
    #
    _1_hop_connect.append(len(np.where(one_C == 1)[0])/2)
    _bb_connect.append(np.sum(np.diagonal(one_L)).astype(np.int32)/2)

print(_1_hop_connect)
print(_bb_connect)
print("last_1_hop:", _1_hop_connect[-1])
print("last_bb:", _bb_connect[-1])

print(np.diagonal(last_L))

plt.plot(steps, _1_hop_connect, color=TYPE2COLOR["NB"], label="1-hop connections")
plt.plot(steps, _bb_connect, color=TYPE2COLOR["CH"], label="Backbone connections")

# 样式
plt.xlabel("Evolution times $k$")
plt.ylabel("The number of connections")
plt.yscale('log')
# plt.yscale('symlog')
# plt.yscale('logit')
# plt.gca().yaxis.set_major_locator(FixedLocator(np.arange(0, 1e5, 1e4)))
# plt.gca().xaxis.set_major_locator(MultipleLocator(1))
# plt.gca().yaxis.set_major_locator(MultipleLocator(3e3))
# plt.gca().yaxis.set_minor_formatter(LogFormatter())
plt.legend()
# plt.grid(True)
# plt.title("Original 1-hop connections V.S. Backbone-based connections")
plt.savefig(f"./vis/connect_num/{log_dir}.png", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/connect_num/{log_dir}.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
