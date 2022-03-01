# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/3/10 21:17

import os
import cv2
import glob
import imageio


log_path = "log_diagram"
gif_name = "diagram.gif"
duration = 2    # 每张图几秒
duration = 1    #
duration = 0.2    #

# 创建一个空列表，用来存源图像
frames = []

# 利用方法append把图片挨个存进列表
imgname_lst = glob.glob(f"./{log_path}/1d_*.png")
imgname_lst.sort(key=lambda x: os.path.getctime(x))
print(imgname_lst)
for image_name in imgname_lst:
    frames.append(imageio.imread(image_name))

# 保存为gif格式的图
imageio.mimsave(f"./gif/{gif_name}", frames, 'GIF', duration=duration)
