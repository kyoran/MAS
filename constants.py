# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/10 11:24

import os
import glob
import shutil

print("loading...")

density_type = "1000"
topology_type = "009"
# algorithm_type = "SAN"
# algorithm_type = "CA"
algorithm_type = "HSBMAS"
# algorithm_type = "RNCA"
# algorithm_type = "RNCHCA"
# algorithm_type = "HSBMAS"

print("\talgorithm type:", algorithm_type)


end_threshold = 1e-2
node_size = 5
priority_sample_num = 2100
# node_size = 20
# node_size = 50

# dataset_path = rf"./dataset/Agents-{density_type}/001log_uniform_200/001log_uniform_200.npy"
dataset_path = glob.glob(rf"./dataset/Agents-{density_type}/{topology_type}*/*.npy")[0]
print("\tdataset from:", dataset_path)

# 中间过程文件的保存地点
topology_fullname_type = dataset_path.replace("\\", "/").split("/")[-2]
print("\ttopology type:", topology_fullname_type)
results_path = rf"./results/Agents-{density_type}/{topology_fullname_type}/{algorithm_type}"
if not os.path.exists(results_path):
    os.makedirs(results_path)
else:
    shutil.rmtree(results_path)
    os.mkdir(results_path)
print("\tlog save path:", results_path)

