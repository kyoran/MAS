# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/13 18:48

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from 汇总数据 import *

from adjustText import adjust_text

plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 15})


import matplotlib.lines as mlines
fig, ax = plt.subplots(1,1,figsize=(10,10), dpi=80)

agent_num = "1000"
methods = ["SAN", "CA", "HSBMAS"]
xs = [1,2,3]
colors = plt.get_cmap("tab10")
colors = colors(np.linspace(0, 1, 10))
colors = dict(zip([f"T{i}" for i in range(1, 11)], colors))
print(colors)
# left_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df.continent, df['1952'])]
# right_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df.continent, df['1957'])]
# klass = ['red' if (y1-y2) < 0 else 'green' for y1, y2 in zip(df['1952'], df['1957'])]

# draw line
# https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
def newline(p1, p2, type_key, label=True):
    ax = plt.gca()
    # l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='red' if p1[1]-p2[1] > 0 else 'green', marker='o', markersize=6)
    if label:
        l = mlines.Line2D(
            [p1[0],p2[0]], [p1[1],p2[1]],
            color=colors[type_key], marker='o', markersize=8, label=type_key
        )
    else:
        l = mlines.Line2D(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=colors[type_key], marker='o', markersize=8
        )
    ax.add_line(l)
    return l


# Vertical Lines
# ymin=20, ymax=4500
ax.vlines(x=xs[0], ymin=20, ymax=4500, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=xs[1], ymin=20, ymax=4500, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=xs[2], ymin=20, ymax=4500, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

# Points
# ax.scatter(y=df['1952'], x=np.repeat(1, df.shape[0]), s=10, color='black', alpha=0.7)
# ax.scatter(y=df['1957'], x=np.repeat(3, df.shape[0]), s=10, color='black', alpha=0.7)

# Line Segmentsand Annotation
ct4each_type = {}   # 每个类别是一个列表[san, ca, hsbmas]存了3个时间
for one_type_idx in range(1, 11):
# for p1, p2, c in zip(df['1952'], df['1957'], df['continent']):
    ct4each_type[f"T{one_type_idx}"] = [
        SAN_ct[0][agent_num][f"{one_type_idx:0>3}"],
        CA_ct[0][agent_num][f"{one_type_idx:0>3}"],
        HSBMAS_ct[0][agent_num][f"{one_type_idx:0>3}"],
    ]
max_y, min_y = 0, 30
min_y_delta, max_y_delta =10, 100

for key, val in ct4each_type.items():
    if max(val) > max_y:
        max_y = max(val)
    if min(val) < min_y:
        min_y = min(val)

def getUpNum(num, limit_num=max_y_delta):
    if num%limit_num!=0:
        num=math.ceil(num/limit_num)*limit_num
    return num

for one_type_key in ct4each_type.keys():
    p1, p2, p3 = ct4each_type[one_type_key]
    newline([xs[0],p1], [xs[1],p2], one_type_key, label=True)
    newline([xs[1],p2], [xs[2],p3], one_type_key, label=False)
    # ax.text(xs[0] - 0.3, p1, one_type_key, horizontalalignment='left', verticalalignment='center',
    #         fontdict={'size': 14, "color": colors[one_type_key], "weight": "bold"})
    t1 = ax.text(xs[0] - 0.05, p1, str(p1), horizontalalignment='right', verticalalignment='center',
            fontdict={'size': 14, 'color': colors[one_type_key]})
    # adjust_text([t1], only_move={'text': 'x'})

    t2 = ax.text(xs[1] + 0.05, p2, str(p2), horizontalalignment='left', verticalalignment='center',
            fontdict={'size': 14, 'color': colors[one_type_key]})
    # adjust_text([t2], only_move={'text': 'x'})

    t3 = ax.text(xs[2] + 0.05, p3, str(p3), horizontalalignment='left', verticalalignment='center',
            fontdict={'size': 14, 'color': colors[one_type_key]})
    # adjust_text([t3], only_move={'text': 'x'})


# 'Before' and 'After' Annotations
ax.text(xs[0], getUpNum(max_y)+200, methods[0], horizontalalignment='center', verticalalignment='center', fontdict={'size':18, 'weight':700})
ax.text(xs[1], getUpNum(max_y)+200, methods[1], horizontalalignment='center', verticalalignment='center', fontdict={'size':18, 'weight':700})
ax.text(xs[2], getUpNum(max_y)+200, methods[2], horizontalalignment='center', verticalalignment='center', fontdict={'size':18, 'weight':700})

# Decoration
# ax.set_title("Slopechart: Comparing GDP Per Capita between 1952 vs 1957", fontdict={'size':22})
ax.set(xlim=(xs[0]-0.2,xs[2]+0.2), ylim=(math.floor(min_y), getUpNum(max_y)), ylabel='Converge time (#CT)')
ax.set_xticks(xs)
ax.xaxis.set_ticklabels([])
# ax.set_xticklabels(["1952", "1957"])
# ax.set_yscale('log', base=0.5)
def forward(x):
    return x**(1/8)
def inverse(x):
    return x**8
ax.set_yscale('function', functions=(forward, inverse))
ax.grid(True)
ax.legend()
plt.yticks(np.arange(20, getUpNum(max_y), 400), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.savefig(f"./vis/ct/{agent_num}-ct.png", bbox_inches='tight', pad_inches=0)
plt.savefig(f"./vis/ct/{agent_num}-ct.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
