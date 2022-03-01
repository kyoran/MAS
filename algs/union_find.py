# !/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by kyoRan on 2021/8/12 20:55


def find(x, pre):
    r = x
    while pre[r] != r:
        r = pre[r]  # 找到前导节点
    i = x
    while i != r:
        j = pre[i]
        pre[i] = r
        i = j
    return r

def join(x, y, pre):
    a = find(x, pre)
    b = find(y, pre)
    if a != b:
        pre[a] = b