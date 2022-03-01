# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/13 18:48

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman')

agents = [
    200,
    400,
    600,
    800,
    1000,
]
types = [1,2,3,4,5,6,7,8,9,10]
# types = [1,2,5,7,8,9]
methods = ["SAN", "CA", "CHCA", "RNCA", "RNCHCA", "HSBMAS"]
methods = ["CA", "CHCA", "RNCA", "RNCHCA", "HSBMAS"]

from 汇总数据 import *

type_keys = []
for one_type in types:
    one_type_key = f"{one_type:0>3d}"
    type_keys.append(one_type_key)
type_keys_en = ["uniform", "ring", "vase", "taiji", "circle", "triangle", "square", "arch", "neat square", "neat radiation"]
type_keys_en_abbr= [f"T{i}" for i in range(1, 11, 1)]
type_num = len(type_keys)

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for one_agent in agents:
    plt.figure(figsize=(14,4))
    plt.title(f"Agents-{one_agent}")
    ax = plt.subplot(111)
    # SAN_cts = [SAN_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]
    CA_cts = [CA_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]
    RNCA_cts = [RNCA_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]
    CHCA_cts = [CHCA_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]
    RNCHCA_cts = [RNCHCA_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]
    HSBMAS_cts = [HSBMAS_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]

    # plt.plot(type_keys, CA_cts, label="CA")
    # plt.plot(type_keys, RNCA_cts, label="RNCA")
    # plt.plot(type_keys, CHCA_cts, label="CHCA")
    # plt.plot(type_keys, RNCHCA_cts, label="RNCHCA")
    # plt.plot(type_keys, HSBMAS_cts, label="HSBMAS")
    x = np.arange(type_num)
    each_width = 0.15  # the width of the bars
    rects1 = ax.bar(x - 2.5*each_width, HSBMAS_cts, each_width, label='HSBMAS')
    rects2 = ax.bar(x - 1.5*each_width, RNCHCA_cts, each_width, label='RNCHCA')
    rects3 = ax.bar(x - 0.5*each_width, CHCA_cts, each_width, label='CHCA')
    rects4 = ax.bar(x + 0.5*each_width, RNCA_cts, each_width, label='RNCA')
    rects5 = ax.bar(x + 1.5*each_width, CA_cts, each_width, label='CA')
    # rects6 = ax.bar(x + 2.5*each_width, SAN_cts, each_width, label='SAN')

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    autolabel(ax, rects3)
    autolabel(ax, rects4)
    autolabel(ax, rects5)
    # autolabel(ax, rects6)
    ax.set_xticks(x)
    ax.set_xticklabels(type_keys_en_abbr)
    ax.legend()
    plt.savefig(f"./vis/ct/{one_agent}-all-cts.png", bbox_inches='tight', pad_inches=0)
plt.show()
