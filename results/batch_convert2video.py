# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/16 10:04


import subprocess

agent_nums = [200, 400, 600, 800, 1000]
# agent_nums = [200]
# topology_ids = [1,2,3,4,5,6,7,8,9,10]
topology_ids = [7]
methods = ["CA"]
# methods = ["HSBMAS"]

for one_agent_num in agent_nums:
    for one_topology_id in topology_ids:
        for one_method in methods:
            # subprocess.call(  # 会出现覆盖的情况
            subprocess.run(
                ["python", "convert2video.py", f"{one_agent_num}", f"{one_topology_id}", f"{one_method}"],
                # shell=True,
            )