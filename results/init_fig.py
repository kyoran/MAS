# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/2/5 10:36

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# plt.style.use(['science','ieee'])

colors = plt.get_cmap("tab10")
colors = colors(np.linspace(0, 1, 10))

plt.figure(figsize=(8, 5))
plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 15})
# plt.figure(figsize=(8,4))

labels = [
    "uniform", "ring", "vase", "taiji", "circle",
    "triangle", "square", "arch",
    "neat square", "neat radiation"
]

labels = [f"T{i}" for i in range(1, 11)]
