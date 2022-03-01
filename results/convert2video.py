# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2020/10/26 23:31

import os
import sys
import cv2
import glob
import shutil

fps = 6

"""需要附加的参数和例子"""
agent_num = sys.argv[1]     # 200
topology_id = sys.argv[2]   # 1
method = sys.argv[3]        # CA


log_path = rf"./Agents-{agent_num}/{topology_id:0>3}*/{method}/*.png"

write_path = f"./video/Agents-{agent_num}/{topology_id:0>3}/"

if not os.path.exists(write_path):
    print("make dir")
    os.makedirs(write_path)
# else:
#     shutil.rmtree(write_path)
#     os.mkdir(write_path)

size = (4000, 4000)

videowriter = cv2.VideoWriter(
    os.path.join(write_path, f"{method}.mp4"),
    cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True
)

imgname_lst = glob.glob(log_path)
imgname_lst.sort(key=lambda x: int(x[x.rindex("\\")+1: x.rindex(".")]))
# print(imgname_lst)

for idx, imgname in enumerate(imgname_lst):
    print(f"\r{log_path} at:", idx+1, "/", len(imgname_lst), os.path.join(write_path, f"{method}.mp4"), end="")

    img = cv2.imread(imgname)
    videowriter.write(img)

print()