# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/10 11:24

import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()

print("loading...")
parser.add_argument('--density_type', default="600", choices=[
    '200', '400', '600', '800', '1000'
])
parser.add_argument('--topology_type', default="001", choices=[
    "001", "002", "003", "004", "005",
    "006", "007", "008", "009", "010",
])
parser.add_argument('--algorithm_type', default="SDB_DSG", choices=[
    'SAN',
    'RSRSP',
    'Motif',
    'CA',
    'RNCHCA',
    'SDB_DSG',
    'HSBMAS',
    'HSBMAS_no_CS',
])
parser.add_argument('--CS', default=False, action='store_true') # 是否加约束集CS
args = parser.parse_args()

#!!!
density_type = args.density_type
topology_type = args.topology_type
algorithm_type = args.algorithm_type
#!!!


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

# 统计数据保存文件
log_file_path = os.path.join(results_path, "0log.txt")
log_file = open(log_file_path, "w+")
