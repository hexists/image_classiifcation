#!/usr/bin/env python3

import os
import sys
import shutil
from collections import defaultdict


src_path = 'nipa_dataset/train'
dst_path_prefix = 'images'

file_list = [fname for fname in os.listdir(src_path) if fname.endswith('.jpg')]

print ("file_list: {}".format(file_list[:10]))

img_cls = defaultdict(list)
for fname in file_list:
    cls = '_'.join(fname.split('_')[:2])
    img_cls[cls].append(fname)


for cls, files in img_cls.items():
    dst_path = '{}/{}'.format(dst_path_prefix, cls)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for fname in files:
        src_file = '{}/{}'.format(src_path, fname)
        dst_file = '{}/{}'.format(dst_path, fname)
        shutil.copy(src_file, dst_file) 
