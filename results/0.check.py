# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/18 21:51

import os
import glob
from 汇总数据 import *

idx = 0

for one_agent_num in ["200", "400", "600", "800", "1000"]:
    one_agent_num_dir = rf"Agents-{one_agent_num}"

    for one_type_id in [f"{i:0>3}" for i in range(1, 11, 1)]:
        one_type_id_dir = rf"{one_type_id}*"

        for one_method in ["CA", "CHCA", "RNCA", "RNCHCA", "HSBMAS"]:

            idx+=1
            print(f"\rnow at: {idx}\t", end="")
            img_list = glob.glob(
                os.path.join(one_agent_num_dir, one_type_id_dir, one_method, "*.png")
            )
            img_list.sort(key=lambda x: int(x[x.rindex("\\")+1: x.rindex(".")]))
            # print(img_list)

            for i, item in enumerate(img_list):
                # 1.看是不是顺序的
                if (i+1) != int(item[item.rindex("\\")+1: item.rindex(".")]):
                    print(f"ERROR at: {one_method}: agent-{one_agent_num}, type-{one_type_id}")


            # 2.看数量是不是和统计的一样
            if len(img_list) == 0:
                if eval(f"{one_method}_ct")[0][one_agent_num][one_type_id] is not None:
                    print(f"ERROR at: {one_method}: agent-{one_agent_num}, type-{one_type_id}")

            else:
                if len(img_list) != eval(f"{one_method}_ct")[0][one_agent_num][one_type_id]:
                    print(f"ERROR at: {one_method}: agent-{one_agent_num}, type-{one_type_id}")

            #
print("\ntotal:", idx)