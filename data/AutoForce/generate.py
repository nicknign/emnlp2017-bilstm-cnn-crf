# -*- coding: utf-8 -*-
# !/usr/bin/env python
import jieba as jb
import sys

reload(sys)
sys.setdefaultencoding('utf8')

with open("origin.txt", "r") as fp:
    all_lines = fp.readlines()

with open("train.txt", "w") as dev_fp:
    for line in all_lines:
        items = line.split(",")
        origin = items[0]
        category = items[1]
        index = 0
        words = jb.cut(origin)
        for word in words:
            dev_fp.write("{} {} O\n".format(index, word))
            index += 1
        dev_fp.write("{}\n\n".format(category))