# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/12 21:16

import datetime
from functools import wraps

def timer(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start_time = datetime.datetime.now()
        print(f"\n*Evolution: {args[4]} start at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        results = f(*args, **kwargs)
        end_time = datetime.datetime.now()
        print(f"**Evolution: {args[4]} End at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, total spend: {(end_time-start_time).seconds / 60: .3f} minutes\n")
        return results
    return decorated