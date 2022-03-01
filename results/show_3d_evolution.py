# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2020/12/15 13:33

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
from 汇总数据 import *

# 过程数据文件的路径
agent_num = 600
method = "HSBMAS"
type_idx = "001"
convergence_time = eval(f"{method}_ct")[0][f"{agent_num}"][type_idx]
more_time = int(convergence_time//6)
steps = list(range(convergence_time+more_time))

all_agents_locations_per_time = [[] for i in range(agent_num)]

location_file_list = glob.glob(
    os.path.join(f"Agents-{agent_num}", f"{type_idx}*", method, "total_locations*")
)
location_file_list.sort(key=lambda x: int(x[x.rindex("_")+1: x.rindex(".")]))

# 设置样式
# plt.style.use("ggplot")
# plt.rc('font', family='Times New Roman')
# plt.rcParams.update({'font.size': 15})
mpl.rcParams['font.family'] = 'SimHei' #设置字体为黑体
mpl.rcParams['axes.unicode_minus'] = False #设置在中文字体是能够正常显示负号（“-”）
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_zlabel("Evolution times $k$")
ax.set_zlabel("收敛时刻 $k$")
ax.view_init(elev=20,    # 仰角
             azim=150    # 方位角
            )
# ax.set_aspect(1. / ax.get_data_ratio())

# 统计轨迹
trajs = []
for i in range(convergence_time):
    total_location = np.load(location_file_list[i])
    # assert agent_num == total_location.shape[0], 'agent num error'
    trajs.append(total_location)



# 不足steps的补齐
for i in range(more_time):
    trajs.append(trajs[-1])

trajs = np.array(trajs)
print(trajs.shape)  # (steps, 1000, 2)

# total_steps = 200
# for i in range(total_steps - trajs.shape[0]):
#     # print(trajs[-1, :, :].shape)
#     trajs = np.vstack((
#         trajs, np.reshape(trajs[-1, :, :], (1, 600, 2))
#     ))
    # print(trajs.shape)

# 绘制轨迹
times = list(range(0, trajs.shape[0], 1))
for aid in range(trajs.shape[1]):   # 400
    ax.plot(trajs[:, aid, 0], trajs[:, aid, 1], times, label=f'agent-{aid}', linewidth=0.6, alpha=0.8)

# 保存图片
plt.savefig(f"./vis/3d_evolution/{method}-{type_idx}_3d_evolution.png", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/3d_evolution/{method}-{type_idx}_3d_evolution.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
