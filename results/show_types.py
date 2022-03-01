# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2020/11/17 18:20

import os
import glob
import numpy as np
import collections
import matplotlib.pyplot as plt
import matplotlib as mpl
from init_fig import *

# plt.rc('font', family='Times New Roman')
mpl.rcParams['font.family'] = 'SimHei' #设置字体为黑体
mpl.rcParams['axes.unicode_minus'] = False #设置在中文字体是能够正常显示负号（“-”）

"""需要修改的路径"""
log_dir = "001log_uniform_600"
# log_dir = "002log_ring_600"
# log_dir = "003log_vase_600"
# log_dir = "004log_taiji_600"
# log_dir = "005log_circle_600"
# log_dir = "006log_triangle_600"
# log_dir = "007log_square_600"
# log_dir = "008log_arch_600"
# log_dir = "009log_neat_square_625"
# log_dir = "010log_neat_radiation_600"

#
log_path = f"Agents-600/{log_dir}/HSBMAS/"

type_filename_lst = glob.glob(f"{log_path}/total_types_*")
type_filename_lst.sort(key=lambda x: int(x[x.rindex("_")+1: x.rindex(".")]))
print(type_filename_lst)

total_types_num = {"CH": [], "DW": [], "GW": [], "NB": []}
steps = list(range(1, len(type_filename_lst)+1))
TYPE2COLOR = {"CH": "orangered", "DW": "seagreen", "GW": "slateblue", "NB": "darkslategray"}

default_keys = {"CH", "DW", "GW", "NB"}
for i in type_filename_lst:
    total_type = np.load(i)
    each_type_nums = dict(collections.Counter(total_type))

    for k, v in each_type_nums.items():
        total_types_num[k].append(v)
    for k in default_keys-each_type_nums.keys():
        total_types_num[k].append(0)

print("CH:", total_types_num["CH"])
print("DW:", total_types_num["DW"])
print("GW:", total_types_num["GW"])
print("NB:", total_types_num["NB"])
print("last CH:", total_types_num["CH"][-1])
print("last DW:", total_types_num["DW"][-1])
print("last GW:", total_types_num["GW"][-1])
print("last NB:", total_types_num["NB"][-1])
plt.plot(steps, total_types_num["CH"], color=TYPE2COLOR["CH"], label="CH")
plt.plot(steps, total_types_num["DW"], color=TYPE2COLOR["DW"], label="DW")
plt.plot(steps, total_types_num["GW"], color=TYPE2COLOR["GW"], label="GW")
plt.plot(steps, total_types_num["NB"], color=TYPE2COLOR["NB"], label="NB")
plt.legend()
# plt.xlabel("Evolution times $k$")
plt.xlabel("收敛时刻 $k$")
plt.ylabel("每种类别智能体的个数")
# plt.title("The number of agents' types (CH, DW, GW, NB) in each step")
plt.savefig(f"./vis/types/{log_dir}.png", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/types/{log_dir}.pdf", bbox_inches='tight', pad_inches=0)
plt.show()

