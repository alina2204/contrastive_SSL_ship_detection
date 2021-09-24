import argparse
import os
import numpy as np
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
import json
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools import mask as coco_mask
import cv2 as cv
import csv
from PIL import Image
from utils import modify_coco, modify_coco_2_cats, convert, change_cat_id

"""
Filter the results obtained by U-Net with the land/sea filtering maps
"""
parser = argparse.ArgumentParser(description='filter results obtained after U-Net')
parser.add_argument('--dataset', help='path to directory containing the predictions. The architecture of folder '
                                      'is dir/0/res.npy, the argument is dir')
parser.add_argument('--filters', help='directory with the filters (sea/land segmentation maps)')
parser.add_argument('--json_dir', help='directory with the json files with COCO ground truth')
parser.add_argument('--filt_litt', default=None, help='if want to filter the littoral, enable this option')
args = parser.parse_args()


if args.filt_litt:
    save_dir = args.dataset[:-1] + '_filter_littoral/'
else:
    save_dir = args.dataset[:-1] + '_filter/'

if not os.path.exists(save_dir):
    print('creating result directory...')
    os.makedirs(save_dir)
json_dir = args.json_dir
with open(args.json_dir + 'coco-s2ships.json', 'r') as json_file:
    data = json_file.read()
data_f = json.loads(data)

coco_new = modify_coco_2_cats(data_f)
with open(json_dir + "targets_json.json", "wt") as file:
    file.write(json.dumps(coco_new))
coco_new_one_cat = modify_coco(data_f)
with open(json_dir + "targets_json_one_cat.json", "wt") as file:
    file.write(json.dumps(coco_new_one_cat))

dir_list = sorted(os.listdir(args.dataset))
mask_list = sorted(os.listdir(args.filters))

name_list = ['01_mask_rome', '02_mask_suez1', '03_mask_suez2', '04_mask_suez3', '05_mask_suez4', '06_mask_suez5',
             '07_mask_suez6', '08_mask_brest1', '09_mask_panama', '10_mask_toulon', '11_mask_marseille',
             '12_mask_portsmouth', '13_mask_rotterdam1', '14_mask_rotterdam2', '15_mask_rotterdam3',
             '16_mask_southampton']
first_row = ['Img Id', 'All TP', 'All FP', 'All FN', 'Sailing ships TP', 'Sailing ships total positives',
             'Moored ships TP', 'Moored ships total positives', 'Small TP', 'Small total positives',
             'Large TP', 'Large total positives']
first_row_conc = ['Img Id', 'Prec', 'recall', 'F1', 'FA rate']
# create csv file to store the final results
concatenated_res = csv.writer(open(save_dir + 'concat_res_filter.csv',
                                   'wt'), lineterminator='\n', )
concatenated_res.writerow(first_row_conc)
concat_res = np.zeros((16, 4))
print(len(dir_list))
for exp in dir_list:
    save_exp_dir = save_dir + '/' + exp + '/'
    if not os.path.exists(save_exp_dir):
        print('creating result directory...')
        os.makedirs(save_exp_dir)
    # for each run, save intermediary results
    csv_no_filter = csv.writer(open(save_exp_dir + 'confusion_mat_filter.csv',
                                    'wt'), lineterminator='\n', )
    csv_no_filter.writerow(first_row)
    exp_list = sorted(os.listdir(args.dataset + '/' + exp))
    # get the prediction dans the name of the img
    for e in exp_list:
        if 'npy' in e:
            id = 1
            for nam in name_list:
                if nam[8:] in e:
                    name = nam[8:]
                    img_id = int(nam[:2])
                    print('img id', img_id)
            preds_img = np.load(args.dataset + '/' + exp + '/' + e)
        else:
            continue
        m, n = preds_img.shape
        key = 'filter'
        for el in mask_list:
            if name in el:  # get name
                mask_img = np.asarray(Image.open(args.filters + el))
                mask_img = mask_img[:m, :n]
                if args.filt_litt:
                    # cut also the littoral
                    mask_img = cv.distanceTransform(mask_img, cv.DIST_L2, 0, dstType=cv.CV_32F)
                    mask_img = np.where(mask_img < 60, 0, 255)
        preds_img = np.where(mask_img == 255, preds_img, 0)

        rgb_img = np.zeros((m, n, 3), dtype=np.uint8)
        # create a new json file with the filtered predictions
        pred_json = {
            "info": {
                "description": "s2ships_predictions",
                "url": "",
                "version": "0.1",
                "year": 2021,
                "contributor": "Theresis/VAR",
                "date_created": "2021/06/07"
            },
            "annotations": [],
            "categories": [
                {"id": 1, "name": "ship", "supercategory": "", "color": "#ffc500", "metadata": {},
                 "keypoint_colors": []},
                {"id": 2, "name": "moored ship", "supercategory": "", "color": "#ffc500", "metadata": {},
                 "keypoint_colors": []}]
        }
        idx = 0
        count2 = 0

        img = rgb_img

        idx = 0

        COLORS = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype='uint8')
        label = ['ship', 'moored ship']

        color = COLORS[0]
        np.save(save_exp_dir + '{a}_predicted_segmentation_mask_{f}.npy'.format(a=name, f=key), preds_img[:, :])
        preds_im = np.where(preds_img[:, :] == 1, 1, 0)
        preds_im = (preds_im * 255).astype(np.uint8)
        out_img = np.where(preds_im[..., None], color, img)
        for cat in range(2, 2):
            color = COLORS[cat - 1]
            preds_im = np.where(preds_img[:, :] == cat, 1, 0)
            preds_im = (preds_im * 255).astype(np.uint8)
            out_img = np.where(preds_im[..., None], color, out_img).astype(np.uint8)

        for cat in range(1, 2):
            color = COLORS[cat - 1]
            preds_im = np.where(preds_img[:, :] == cat, 1, 0)
            preds_im = (preds_im * 255).astype(np.uint8)

            # get components from mask
            n_comp, labels, stats, centroids = cv.connectedComponentsWithStats(preds_im)
            for n in range(1, n_comp):
                componentMask = (labels == n).astype("uint8")
                mask_json_preds = coco_mask.encode(np.asfortranarray(componentMask, dtype=np.uint8))

                mask_json_preds['counts'] = mask_json_preds['counts'].decode('utf8')

                pred_json["annotations"].append({"id": idx,
                                                 "image_id": img_id,
                                                 "category_id": cat,
                                                 "segmentation": mask_json_preds,
                                                 "score": 1,
                                                 "iscrowd": 0,
                                                 "area": int(stats[n, 4]),
                                                 "bbox": list(stats[n, 0:4])
                                                 })

                idx += 1
                x = stats[n, cv.CC_STAT_LEFT]
                y = stats[n, cv.CC_STAT_TOP]
                w = stats[n, cv.CC_STAT_WIDTH]
                h = stats[n, cv.CC_STAT_HEIGHT]
                cv.rectangle(out_img, (x, y), (x + w, y + h), tuple(color.tolist()), 1)

        cv.imwrite(save_exp_dir + 'visualization_{e}_{f}.png'.format(e=name, f=key),
                   out_img)

        if len(pred_json["annotations"]) == 0:
            # if no objects detected, continue
            print('STOP : no annotation found for this img')
            continue

        # perform evaluation
        with open(json_dir + "preds_json.json", "wt") as file:
            file.write(json.dumps(pred_json["annotations"], default=convert))
        cocoGt = COCO(json_dir + "targets_json_one_cat.json")
        cocoDt = cocoGt.loadRes(json_dir + "preds_json.json")
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")

        per_size_tp_list = []
        per_size_nb_pos = []
        conf_list = []

        cocoEval.params.useCats = 0
        cocoEval.params.imgIds = [img_id]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # get metrics
        conf_list.append(cocoEval.get_confusion_matrix())
        tp_small, nb_positives_small = cocoEval.get_tp_pos_small()
        tp_large, nb_positives_large = cocoEval.get_tp_pos_large()
        per_size_tp_list.append(tp_small)
        per_size_tp_list.append(tp_large)
        per_size_nb_pos.append(nb_positives_small)
        per_size_nb_pos.append(nb_positives_large)
        # per class evaluation
        per_class_tp_list = []
        per_class_nb_pos = []

        # perform per class evaluation (moored and sailing ships)
        cocoGt = COCO(json_dir + "targets_json.json")
        for cat in range(1, 3):
            with open(json_dir + 'preds_json.json', 'r') as json_file:
                data = json_file.read()
            data_f = json.loads(data)
            new_coco = change_cat_id(data_f, cat)
            with open(json_dir + "preds_json_modified.json", "wt") as file:
                file.write(json.dumps(new_coco, default=convert))
            cocoDt = cocoGt.loadRes(json_dir + "preds_json_modified.json")
            cocoEval = COCOeval(cocoGt, cocoDt, "segm")
            cocoEval.params.imgIds = [img_id]
            cocoEval.params.catIds = [cat]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            tp, nb_positives = cocoEval.get_tp_pos()
            per_class_tp_list.append(tp)
            per_class_nb_pos.append(nb_positives)

        # write csv file
        row_bis = [name_list[img_id - 1]]
        conf_mat = conf_list[-1].tolist()
        row_bis.extend(conf_mat)
        row_bis.append(per_class_tp_list[0])
        row_bis.append(per_class_nb_pos[0])

        row_bis.append(per_class_tp_list[1])
        row_bis.append(per_class_nb_pos[1])

        row_bis.append(per_size_tp_list[0])
        row_bis.append(per_size_nb_pos[0])

        row_bis.append(per_size_tp_list[1])
        row_bis.append(per_size_nb_pos[1])
        csv_no_filter.writerow(row_bis)

        # prepare mean metrics, averaged over all runs
        precision = conf_mat[0] / (conf_mat[0] + conf_mat[1])
        recall = conf_mat[0] / (conf_mat[0] + conf_mat[2])
        F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        FA = conf_mat[1] / (17.80 * 9.30)
        add_el = [precision, recall, F1, FA]
        for j in range(concat_res.shape[1]):
            concat_res[img_id - 1][j] = concat_res[img_id - 1][j] + add_el[j]

# write final csv file by averaging the results
for img_id in range(concat_res.shape[0]):
    for j in range(concat_res.shape[1]):
        concat_res[img_id][j] = concat_res[img_id][j] / len(dir_list)
    row = [name_list[img_id]]
    row.extend(list(concat_res[img_id]))

    concatenated_res.writerow(row)
