# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/2/12 21:49

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from init_fig import *

"""需要修改的路径"""
log_paths = [
    "./Agents-200/001log_uniform_200/HSBMAS/",
    "./Agents-400/001log_uniform_400/HSBMAS/",
    "./Agents-600/001log_uniform_600/HSBMAS/",
    "./Agents-800/001log_uniform_800/HSBMAS/",
    "./Agents-1000/001log_uniform_1000/HSBMAS/",
]
labels = [
    r"T1 ($\rho=2$)",
    r"T1 ($\rho=4$)",
    r"T1 ($\rho=6$)",
    r"T1 ($\rho=8$)",
    r"T1 ($\rho=10$)",
]

total_steps = []
total_eigenvalues = []
max_len = 0
for one_log_path in log_paths:
    steps = []
    eigenvalues = []
    with open(f"{one_log_path}/eigenvalue.txt", "r") as file:
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

# max_len = 1000
# 补充，保证数量一样
print("max_len:", max_len)
for i in range(len(labels)):
    tmp_len = len(total_eigenvalues[i])
    if tmp_len < max_len:
        total_eigenvalues[i] += [total_eigenvalues[i][-1]] * (max_len-tmp_len)
    else:
        total_eigenvalues[i] = total_eigenvalues[i][:max_len]


for each_log_id in range(len(log_paths)):
    plt.plot(
        list(range(0, max_len, 1)),
        total_eigenvalues[each_log_id],
        color=colors[each_log_id],
        label=labels[each_log_id].capitalize()
    )
plt.yscale('log')

plt.xlabel("Evolution times $k$")
plt.ylabel("The second smallest eigenvalue $\lambda_2$")
plt.legend()
plt.savefig(f"./vis/eigenvalue/density_eigenvals.pdf", bbox_inches='tight', pad_inches=0)
# plt.savefig(f"./eigenvalue/total_eigenvalues_{method}_detail.svg")
plt.savefig(f"./vis/eigenvalue/density_eigenvals.svg", bbox_inches='tight', pad_inches=0)
plt.show()


