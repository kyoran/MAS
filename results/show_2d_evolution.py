# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2020/11/24 15:02


import os
import glob
import numpy as np
import collections
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator
from 汇总数据 import *

plt.rc('font', family='Times New Roman')

"""需要修改的路径"""
agent_num = 600
method = "HSBMAS"
type_idx = "010"
convergence_time = eval(f"{method}_ct")[0][f"{agent_num}"][type_idx]
more_time = int(convergence_time//6)
steps = list(range(convergence_time+more_time))

all_agents_locations_per_time = [[] for i in range(agent_num)]

location_file_list = glob.glob(
    os.path.join(f"Agents-{agent_num}", f"{type_idx}*", method, "total_locations*")
)
location_file_list.sort(key=lambda x: int(x[x.rindex("_")+1: x.rindex(".")]))
print(location_file_list)

assert convergence_time == len(location_file_list)

for i in range(convergence_time):
    total_location = np.load(location_file_list[i])
    for aid in range(agent_num):
        all_agents_locations_per_time[aid].append(total_location[aid])

# 补充more_time个，体现出在一起
for i in range(more_time):
    for aid in range(agent_num):
        all_agents_locations_per_time[aid].append(
            all_agents_locations_per_time[aid][-1]
        )

all_agents_locations_per_time = np.array(all_agents_locations_per_time)

# X
plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 15})
plt.figure()
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(1))
for i in range(agent_num):
    plt.plot(steps, all_agents_locations_per_time[i, :, 0], label=f"agent-{i}")
plt.xlabel("Evolution times $k$")
plt.xlim([0, convergence_time+more_time])
plt.ylabel("X")
plt.ylim([-5,5])
plt.title("")
# plt.savefig(f"./2d_evolution/{log_dir}_x.svg")
plt.savefig(f"./vis/2d_evolution/{method}-{type_idx}_x.png", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/2d_evolution/{method}-{type_idx}_x.pdf", bbox_inches='tight', pad_inches=0)



# Y
plt.figure()
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(1))
for i in range(agent_num):
    plt.plot(steps, all_agents_locations_per_time[i, :, 1], label=f"agent-{i}")
plt.xlabel("Evolution times $k$")
plt.xlim([0, convergence_time+more_time])
plt.ylabel("Y")
plt.ylim([-5,5])
plt.title("")
# plt.savefig(f"./2d_evolution/{log_dir}_y.svg")
plt.savefig(f"./vis/2d_evolution/{method}-{type_idx}_y.png", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/2d_evolution/{method}-{type_idx}_y.pdf", bbox_inches='tight', pad_inches=0)
plt.show()


