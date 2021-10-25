import os
import glob
import re
import cv2
import numpy as np
from PIL import Image
import json
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools import mask as coco_mask
import csv
import argparse
from utils import modify_coco, modify_coco_2_cats, convert, change_cat_id

"""
Baseline BL-NDWI
"""

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='./', help='saving directory')

args = parser.parse_args()


def eurosat_norm(b, index):
    """

    :param b: band to be normalized (2D array)
    :param index: index of the given band (int between 0 and 11)
    :return: normalized band
    """
    mean_EuroSAT = [0.44929576, 0.4386203, 0.45689246, 0.45665017, 0.47687784, 0.44870496,
                    0.44587377, 0.44572416, 0.4612574, 0.3974199, 0.47645673, 0.45139566]
    std_EuroSAT = [0.2883096, 0.29738334, 0.29341888, 0.3096154, 0.29744068, 0.28400135,
                   0.2871275, 0.28741345, 0.27953532, 0.22587752, 0.302901, 0.28648832]
    return (b - mean_EuroSAT[index]) / std_EuroSAT[index]


def norm_band_bis(b):
    """
        :param b: band to be normalized (2D array)
        :return: clipped band with values between 0 and 1
        """
    mi, ma = np.nanpercentile(b, (3, 97))
    clipped = np.clip((b - mi) / (ma - mi), 0, 1)

    return clipped


def norm_band(b):
    """
            :param b: band to be normalized (2D array)
            :return: clipped band with values between 0 and 255 and type uint8
            """
    mi, ma = np.nanpercentile(b, (3, 97))
    clipped = np.clip((b - mi) / (ma - mi + 0.00001), 0, 1) * 255
    return clipped.astype(np.uint8)


def evaluate_best_thresh(rgb_img, preds_img, json_dir, save_dir, img_id, thresh, csv_file):
    """

        :param rgb_img: rgb version of the predicted image (array)
        :param preds_img: numpy array of predicted mask (array)
        :param json_dir: path of json annotations files (string)
        :param save_dir: saving directory (string)
        :param img_id: id of the tested image (for S2-SHIP dataset) (int)
        :param thresh: threshold (float)
        :param csv_file: csv object
        :return: writes in a csv file the metrics obtained on each test image + writes the predicted image
        """

    name_list = ['01_mask_rome', '02_mask_suez1', '03_mask_suez2', '04_mask_suez3', '05_mask_suez4', '06_mask_suez5',
                 '07_mask_suez6', '08_mask_brest1', '09_mask_panama', '10_mask_toulon', '11_mask_marseille',
                 '12_mask_portsmouth', '13_mask_rotterdam1', '14_mask_rotterdam2', '15_mask_rotterdam3',
                 '16_mask_southampton']
    np.save(save_dir + '{n}_pred_{t}.npy'.format(n=name_list[img_id - 1], t=thresh), preds_img)
    num = 1
    conf_list = []

    # load and prepare annotations files
    with open(json_dir + 'coco-s2ships.json', 'r') as json_file:
        data = json_file.read()
    data_f = json.loads(data)
    coco_new = modify_coco_2_cats(data_f)
    with open(json_dir + "targets_json.json", "wt") as file:
        file.write(json.dumps(coco_new))
    coco_new_one_cat = modify_coco(data_f)
    with open(json_dir + "targets_json_one_cat.json", "wt") as file:
        file.write(json.dumps(coco_new_one_cat))
    rec_list = []
    prec_list = []
    pred_json = {
        "info": {
            "description": "s2ships_predictions",
            "url": "",
            "version": "0.1",
            "year": 2021,
            "contributor": "Alina",
            "date_created": "2021/06/07"
        },
        "annotations": [],
        "categories": [
            {"id": 1, "name": "ship", "supercategory": "", "color": "#ffc500", "metadata": {},
             "keypoint_colors": []},
            {"id": 2, "name": "moored ship", "supercategory": "", "color": "#ffc500", "metadata": {},
             "keypoint_colors": []}]
    }

    # convert predicted masks into coco annotation file
    for i in range(num):
        img = rgb_img.astype(np.uint8)
        color = np.array([255, 0, 0], dtype='uint8')
        idx = 0
        pred = (preds_img * 255).astype(np.uint8)
        masked_img = np.where(pred[..., None], color, img).astype(np.uint8)
        out = cv2.addWeighted(img, 0.7, masked_img, 0.7, 0)
        n_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(pred)
        for n in range(1, n_comp):
            componentMask = (labels == n).astype("uint8")
            mask_json_preds = coco_mask.encode(np.asfortranarray(componentMask, dtype=np.uint8))

            mask_json_preds['counts'] = mask_json_preds['counts'].decode('utf8')

            pred_json["annotations"].append({"id": idx,
                                             "image_id": img_id,
                                             "category_id": 1,
                                             "segmentation": mask_json_preds,
                                             "score": 1,
                                             "iscrowd": 0,
                                             "area": int(stats[n, 4]),
                                             "bbox": list(stats[n, 0:4])
                                             })

            idx += 1
            x = stats[n, cv2.CC_STAT_LEFT]
            y = stats[n, cv2.CC_STAT_TOP]
            w = stats[n, cv2.CC_STAT_WIDTH]
            h = stats[n, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.imwrite(save_dir + '{n}_visualization_thresh_{t}.png'.format(n=name_list[img_id - 1], t=round(thresh, 2)),
                    out)

        if len(pred_json["annotations"]) == 0:
            # if no annotations found, set all metrics to 0
            rec_list.append(0)
            prec_list.append(0)
            conf_list.append([])
            continue

        # prepare json files for the evaluation
        with open(json_dir + "preds_json.json", "wt") as file:
            file.write(json.dumps(pred_json["annotations"], default=convert))

        with open(json_dir + "preds_json.json", "wt") as file:
            file.write(json.dumps(pred_json["annotations"], default=convert))
        cocoGt = COCO(json_dir + "targets_json_one_cat.json")
        cocoDt = cocoGt.loadRes(json_dir + "preds_json.json")
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")

        per_size_tp_list = []
        per_size_nb_pos = []
        conf_list = []

        # evaluation for all kind of ships
        cocoEval.params.useCats = 0
        cocoEval.params.imgIds = [img_id]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        conf_list.append(cocoEval.get_confusion_matrix())

        # per size results
        tp_small, nb_positives_small = cocoEval.get_tp_pos_small()
        tp_large, nb_positives_large = cocoEval.get_tp_pos_large()
        per_size_tp_list.append(tp_small)
        per_size_tp_list.append(tp_large)
        per_size_nb_pos.append(nb_positives_small)
        per_size_nb_pos.append(nb_positives_large)

        # per class evaluation
        per_class_tp_list = []
        per_class_nb_pos = []
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

        # write results in csv file
        row_bis = [name_list[img_id - 1]]
        row_bis.extend(conf_list[-1].tolist())
        row_bis.append(per_class_tp_list[0])
        row_bis.append(per_class_nb_pos[0])

        row_bis.append(per_class_tp_list[1])
        row_bis.append(per_class_nb_pos[1])

        row_bis.append(per_size_tp_list[0])
        row_bis.append(per_size_nb_pos[0])

        row_bis.append(per_size_tp_list[1])
        row_bis.append(per_size_nb_pos[1])

        csv_file.writerow(row_bis)


def compute_seg(ndwi_img, thresh, mask_img):
    """
    :param ndwi_img: NDWI image to be threshold (2D array)
    :param thresh: threshold to apply to NDWI img
    :param mask_img: sea/land segmentation mask
    :return: NDWI thresholded & mask filtered image
    """
    ndwi_seg_img = np.where(ndwi_img < (thresh * 255), 1, 0).astype(np.uint8) * 255
    ndwi_seg_img = np.where(mask_img == 255, ndwi_seg_img, 0).astype(np.uint8) * 255
    return ndwi_seg_img


def seg_items(map, kernel):
    """
    :param map: predicted mask to analyse (array)
    :param kernel: kernel for morphology functions
    :return: mask with filtered and segmented components
    """
    num_comp, comp, stats, centers = cv2.connectedComponentsWithStats(map)
    good_comp = np.where(np.logical_and(stats[:, cv2.CC_STAT_WIDTH] < 70, stats[:, cv2.CC_STAT_HEIGHT] < 70))
    good_comp = list(good_comp[0])
    map = np.zeros_like(map, np.uint8)
    for idx in good_comp:
        map[comp == idx] = 255
    opening = cv2.morphologyEx(map, cv2.MORPH_OPEN, kernel, iterations=1)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5, dstType=cv2.CV_32F)
    ret, sure_fg = cv2.threshold(dist_transform, 0, 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    dispmarkers = np.clip(markers.copy() * (255 / ret), 0, 255).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(map, cv2.COLOR_GRAY2BGR), markers)
    dispmarkers = np.clip(markers.copy() * (255 / ret), 0, 255).astype(np.uint8)
    markers[markers == 1] = 0
    return (markers > 0).astype(np.uint8)


name_list = ['01_mask_rome', '02_mask_suez1', '03_mask_suez2', '04_mask_suez3', '05_mask_suez4', '06_mask_suez5',
             '07_mask_suez6', '08_mask_brest1', '09_mask_panama', '10_mask_toulon', '11_mask_marseille',
             '12_mask_portsmouth', '13_mask_rotterdam1', '14_mask_rotterdam2', '15_mask_rotterdam3',
             '16_mask_southampton']
dataset_full_path = args.save_dir + "/s2ships/dataset_full/"
dir_list = sorted(os.listdir(dataset_full_path))
dataset_path = args.save_dir + "/S2ships/"
save_dir = args.save_dir + "/u_net_results/results_baseline/"
num_thresh = 10
img_ids = []

# get img shape
for dir in glob.glob(dataset_path + "/*"):
    if os.path.isdir(dir):
        tile_name = os.path.basename(dir)
        if tile_name in ["zips", "docker", "ndwi"]:
            continue
        band_files = glob.glob(dir + "/*Sentinel-2_L2A_B*")
        bands = {}
        for b in band_files:
            band_id = re.search("_B(\d.)_", b).group(1)
            bimg = np.array(Image.open(b))
            bands[band_id] = bimg
        m, n = bimg.shape[0], bimg.shape[1]
        break

preds_img_no_filter = np.zeros((num_thresh, m, n))
thresh_list = np.linspace(0.35, 0.95, num=num_thresh)

# initialize csv files to store the results
for thresh in thresh_list:
    first_row = ['Img Id', 'All TP', 'All FP', 'All FN', 'Sailing ships TP',
                 'Sailing ships total positives',
                 'Moored ships TP', 'Moored ships total positives', 'Small TP', 'Small total positives',
                 'Large TP', 'Large total positives']
    csvfile_no_filt = open(save_dir + 'confusion_matrix_nf_rf_thresh_{}.csv'.format(round(thresh, 2)), 'wt')
    writer_nf = csv.writer(csvfile_no_filt, lineterminator='\n', )
    writer_nf.writerow(first_row)

# dictionary to store the data
data = {"img": [], "label": [], "mask": [], "rgb": [], "name": []}
bands_indices = [1, 2, 7, 10, 11]
nb_bands = len(bands_indices) + 1

mask_path = args.save_dir + "/s2ships/water_mask/"
mask_dir_list = sorted(os.listdir(mask_path))

# get and store the data
for name in name_list:
    for el in dir_list:
        if name in el:
            img_and_label = np.load(dataset_full_path + el, allow_pickle=True)
            for elmt in mask_dir_list:
                if name[8:] in elmt:
                    mask_img = np.asarray(Image.open(mask_path + elmt))
                    mask_img = mask_img[:m, :n]
            sample = img_and_label.item().get("data")
            label = img_and_label.item().get("label")

            # noramlize img and calculate NDWI
            ndwi_img = (sample[..., 2].astype(np.float) - sample[..., 7]) / \
                       (sample[..., 2].astype(np.float) + sample[..., 7] + 0.000001)
            ndwi_img = norm_band_bis(ndwi_img)
            bands_5_img = np.zeros((m, n, len(bands_indices) + 1))
            for i, e in enumerate(bands_indices):
                bands_5_img[..., i] = eurosat_norm(norm_band_bis(sample[..., e]), e)
            bands_5_img[..., -1] = ndwi_img

            rgb_img = np.zeros((ndwi_img.shape[0], ndwi_img.shape[1], 3), dtype=np.uint8)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            rgb_img[:, :, 0] = clahe.apply(norm_band(sample[:, :, 1]))
            rgb_img[:, :, 1] = clahe.apply(norm_band(sample[:, :, 2]))
            rgb_img[:, :, 2] = clahe.apply(norm_band(sample[:, :, 3]))

            data["img"].append(bands_5_img)
            data["label"].append(label)
            data["name"].append(name)
            data["rgb"].append(rgb_img)
            data["mask"].append(mask_img)

# apply to each image different threshold and store the results in csv file
for i in range(len(data["name"])):
    sample = data["img"][i]
    label = data["label"][i]
    rgb_img = data["rgb"][i]
    name_img = data["name"][i]
    mask_img = data["mask"][i]
    img_id = i + 1

    index = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for thresh in thresh_list:
        csvfile_no_filt = open(save_dir + 'confusion_matrix_nf_rf_thresh_{}.csv'.format(round(thresh, 2)),
                               'a')
        writer_nf = csv.writer(csvfile_no_filt, lineterminator='\n', )
        print('predicting with threshold ', round(thresh, 2))

        # apply threshold
        ndwi_seg = compute_seg((255 * sample[..., -1]).astype(np.uint8), thresh, mask_img)

        # filter the components
        ndwi_seg_img_no_filter = seg_items(ndwi_seg, kernel)
        preds_img_no_filter[index, :, :] = ndwi_seg_img_no_filter

        # evaluate the results
        evaluate_best_thresh(rgb_img, preds_img_no_filter[index], dataset_path, save_dir, img_id, thresh, writer_nf)
        csvfile_no_filt.close()
