# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2020/11/15 15:31

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from init_fig import *


"""需要修改的路径"""
agent_num = 600
real_agent_num = 625
method = "HSBMAS"
# method = "SAN"
log_paths = [
    f"001log_uniform_{agent_num}",          f"002log_ring_{agent_num}",
    f"003log_vase_{agent_num}",             f"004log_taiji_{agent_num}",
    f"005log_circle_{agent_num}",           f"006log_triangle_{agent_num}",
    f"007log_square_{agent_num}",           f"008log_arch_{agent_num}",
    f"009log_neat_square_{real_agent_num}", f"010log_neat_radiation_{agent_num}"
]

total_steps = []
total_eigenvalues = []
max_len = 0
for one_log_path in log_paths:
    steps = []
    eigenvalues = []
    with open(f"./Agents-{agent_num}/{one_log_path}/{method}/eigenvalue.txt", "r") as file:
        lines = file.readlines()
        lines = list(reversed(lines))
        for one_line in lines:
            one_line = one_line[:-1]
            step, tmp_eigenvalue = one_line.split(",")
            eigenvalues.append(eval(tmp_eigenvalue).real)
            steps.append(eval(step))
            if step == "1":
                break
    steps = list(reversed(steps))
    eigenvalues = list(reversed(eigenvalues))
    total_steps.append(steps)
    total_eigenvalues.append(eigenvalues)
    if len(eigenvalues) > max_len:
        max_len = len(eigenvalues)

max_len = 1000
max_len = 600       # SAN
max_len = 200       # HSBMAS
# 补充，保证数量一样
print("max_len:", max_len)
for i in range(len(log_paths)):
    tmp_len = len(total_eigenvalues[i])
    if tmp_len < max_len:
        total_eigenvalues[i] += [total_eigenvalues[i][-1]] * (max_len-tmp_len)
    else:
        total_eigenvalues[i] = total_eigenvalues[i][:max_len]


for each_log_id in range(len(log_paths)):
    plt.plot(
        list(range(1, max_len+1, 1)),
        total_eigenvalues[each_log_id],
        color=colors[each_log_id],
        label=labels[each_log_id].capitalize()
    )
plt.yscale('log')
plt.xlabel("Evolution times $k$")
plt.ylabel("The second smallest eigenvalue $\lambda_2$")
plt.legend()
plt.savefig(f"./vis/eigenvalue/total_eigenvalues_{method}.pdf", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/eigenvalue/total_eigenvalues_{method}.svg")
plt.savefig(f"./vis/eigenvalue/total_eigenvalues_{method}.png", dpi=500)
plt.show()

