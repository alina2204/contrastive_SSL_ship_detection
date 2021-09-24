import numpy as np
import json
import argparse
import os
from cocoapi.PythonAPI.pycocotools.coco import COCO

"""
Generate masks from coco annotations
"""

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', help='train dataset path')
parser.add_argument('--nb_classes', type=int, help='number of ship classes, must be 1 or 2')
args = parser.parse_args()

with open(args.save_dir + 's2ships/coco-s2ships.json', 'r') as json_file:
    data = json_file.read()
data_f = json.loads(data)

coco = COCO(args.save_dir + 's2ships/coco-s2ships.json')
imgIds = coco.getImgIds()
for id in imgIds:
    f_name = data_f['images'][id - 1]['file_name']
    print(f_name)
    annids = coco.getAnnIds([id])
    anns = coco.loadAnns(annids)
    if args.nb_classes == 2:
        mask = np.zeros((938, 1783, 2))
        for ann in anns:
            if ann['category_id'] == 4:
                mask[:, :, 1] += coco.annToMask(ann)
            else:
                mask[:, :, 0] += coco.annToMask(ann)
    if args.nb_classes == 1:
        mask = np.zeros((938, 1783, 1))
        for ann in anns:
            mask[:, :, 0] += coco.annToMask(ann)
    if id < 10:
        id = '0' + str(id)
    save_dir_path = args.save_dir + 's2ships/s2ships_labels_npy/'
    if not os.path.exists(save_dir_path):
        print('creating result directory...')
        os.makedirs(save_dir_path)
    np.save(save_dir_path + '{id}_mask_{e}'.format(id=id, e=f_name), mask)
