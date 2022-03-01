# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/13 18:48

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from 汇总数据 import *


plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 15})
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
ax.view_init(elev=30, azim=60)
ax.set_box_aspect(aspect = (6,4,2))

colors = plt.cm.viridis(np.linspace(0, 1, 10))

width = depth = 0.3

yticks_labels = ['Agents-200', 'Agents-400', 'Agents-600', 'Agents-800', 'Agents-1000']
yticks = np.arange(len(yticks_labels)) #+ depth/2  # the label locations
xticks_labels = [f'T{i}' for i in range(1, 11)]
xticks = [index+width/2 for index in range(len(xticks_labels))]

yticks_dict = dict(zip(yticks_labels, yticks))
xticks_dict = dict(zip(xticks_labels, xticks))

# ax1.set_ylabel(r'Dataset')
ax.set_yticks(yticks)
ax.set_yticklabels(yticks_labels)
ax.set_zlabel("Evolution times $k$")
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))

# ax.set_xlabel(r'$\varepsilon$')
ax.set_xticks(xticks)
ax.set_xticklabels(xticks_labels)

_x = np.array(xticks)
_y = np.array(yticks)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    print(rects)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # for rect in rects:
        # height = rect.get_height()
        # ax.annotate('{}'.format(height),
        #             xy=(rect.get_x() + rect.get_width() / 2, height),
        #             xytext=(0, 2),  # 3 points vertical offset
        #             textcoords="offset points",
        #             ha='center', va='bottom')
        # ax.text(x, y, z, '%s' % (label), size=20, zorder=1, color='k')


for each_agent_num in [num[num.rindex("-")+1:] for num in yticks_labels]:
    # print(each_agent_num)
    for one_x in xticks_labels:
        each_type = f"{one_x[1:]:0>3}"
        index = np.where(
            (y==yticks_dict[f'Agents-{each_agent_num}']) & (x==xticks_dict[one_x])
        )

        x_tmp = x[index]
        y_tmp = y[index]
        print("x_tmp:", x_tmp)
        print("y_tmp:", y_tmp)

        # HSBMAS
        top_HSBMAS = HSBMAS_ct[0][f"{each_agent_num}"][each_type]
        print(top_HSBMAS)
        ax.bar3d(
            x_tmp, y_tmp-depth / 2,
            bottom, width / 2, depth,
            # [top_HSBMAS]*len(x_tmp),
            top_HSBMAS,
            shade=True, color=colors[8],
            label='HSBMAS',
        )
        ax.text(x_tmp[0], (y_tmp-depth / 2)[0], top_HSBMAS, f"{top_HSBMAS}", color="k", zorder=5, horizontalalignment='left', verticalalignment='bottom')
        # CA
        top_CA = CA_ct[0][f"{each_agent_num}"][each_type]
        ax.bar3d(
            x_tmp- width/2, y_tmp-depth / 2,
            bottom, width / 2, depth,
            # [top_CA] * len(x_tmp),
            top_CA,
            shade=True, color=colors[2],
            label='CA',
        )
        ax.text((x_tmp- width/2)[0], (y_tmp-depth / 2)[0], top_CA, f"{top_CA}", color="k", zorder=5, horizontalalignment='left', verticalalignment='bottom')


# plt.savefig(f"{figname}.pdf", bbox_inches='tight', pad_inches=0)


# for one_agent in agents:
#     CA_cts = [CA_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]
#     # RNCA_cts = [RNCA_ct[0]["200"][one_type_key] for one_type_key in type_keys]
#     # CHCA_cts = [CHCA_ct[0]["200"][one_type_key] for one_type_key in type_keys]
#     # RNCHCA_cts = [RNCHCA_ct[0]["200"][one_type_key] for one_type_key in type_keys]
#     HSBMAS_cts = [HSBMAS_ct[0][f"{one_agent}"][one_type_key] for one_type_key in type_keys]
#
#     # plt.plot(type_keys, CA_cts, label="CA")
#     # plt.plot(type_keys, RNCA_cts, label="RNCA")
#     # plt.plot(type_keys, CHCA_cts, label="CHCA")
#     # plt.plot(type_keys, RNCHCA_cts, label="RNCHCA")
#     # plt.plot(type_keys, HSBMAS_cts, label="HSBMAS")
#     x = np.arange(type_num)
#     width = 0.4  # the width of the bars
#     rects1 = ax.bar(x - width / 2, HSBMAS_cts, width, label='HSBMAS')
#     rects2 = ax.bar(x + width / 2, CA_cts, width, label='CA')
#
#     autolabel(ax, rects1)
#     autolabel(ax, rects2)
#     ax.set_xticks(x)
#     ax.set_xticklabels(type_keys_en_abbr)
#     ax.legend()
#
plt.show()
