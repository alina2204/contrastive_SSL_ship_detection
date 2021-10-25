import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchsat.transforms.transforms_seg as T_seg
from datasets import s2ship_patch, s2ship_patch_test
import custom_transforms
from Classifiers import UNet
from sklearn.metrics import jaccard_score as jaccard
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from utils import FocalLoss_b, modify_coco, modify_coco_2_cats, change_cat_id
from Resnet_torchsat import resnet50  # resnet adapting imagenet pretrained weights from 3 channels to n
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
import torch.onnx
import json
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools import mask as coco_mask
import cv2 as cv
import torch.nn.functional as F
import csv
from Models_ssl import Moco18_sat

"""
U-Net based on ResNet encoder pipeline for ship detection in S2-SHIPS dataset 
(insired from Torchsat tuto : https://github.com/sshuair/torchsat)
2 experiments :
- leave-one-out testing, see "main_all_img" function
- vary the number of training image, see  "main_vary_img" function
"""


def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device):
    """
    :param epoch: current epoch (int)
    :param dataloader: training dataloader
    :param model: U-Net model (pytorch model)
    :param criterion: loss function
    :param optimizer: optimizer
    :param device: cpu or gpu device
    :return: train one epoch
    """
    print('train epoch {}'.format(epoch))
    model.train()
    loss_list = []

    # for each batch in dataloader
    for idx, (inputs, targets) in tqdm(enumerate(dataloader)):
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    print('train-epoch:{}, loss: {:5.3}'.format(epoch, loss.item()))


def evalidation(epoch, dataloader, model, criterion, device):
    """
    :param epoch: current epoch (int)
    :param dataloader: validation dataloader
    :param model: U-Net model (pytorch model)
    :param criterion: loss function
    :param device: cpu or gpu device
    :return: validate the training epoch by giving a per pixel evaluation (jaccard) + loss
    """
    print('\neval epoch {}'.format(epoch))
    model.eval()
    mean_loss = []
    mean_jaccard = []
    with torch.no_grad():
        for idx, (inputs, targets) in tqdm(enumerate(dataloader)):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            outputs = F.softmax(outputs, dim=1)  # 1 7 256 256
            outputs = torch.argmax(outputs, dim=1)  # 1 1 256 256
            preds = torch.squeeze(outputs)  # 256 256
            mean_loss.append(loss.item())
            target = targets.cpu().numpy()
            pred = preds.cpu().numpy()
            a, _, _ = target.shape
            # calculate pixelwise jaccard score depending on if we have 2 ship classes or one only
            for h in range(a):
                jac1 = jaccard(target[h].reshape(-1), pred[h].reshape(-1), labels=[1], average='micro', zero_division=0)
                if args.num_classes == 2:
                    jac2 = jac1
                else:
                    jac2 = jaccard(target[h].reshape(-1), pred[h].reshape(-1), labels=[2], average='micro',
                                   zero_division=0)
                mean_jaccard.append((jac1 + jac2) / 2)
    print('mean jaccard', np.nanmean(mean_jaccard))
    print('mean loss', np.nanmean(mean_loss))
    return np.mean(mean_loss), np.mean(mean_jaccard)


def convert(o):
    if isinstance(o, np.int32):
        return int(o)
    if isinstance(o, np.int64):
        return int(o)
    print('wrong type : ', type(o))
    raise TypeError


def predictions(dataset_test, test_ind, model, device, csv_no_filter, it, train_img, concat_res):
    """
    :param dataset_test: test dataset
    :param test_ind: index of the image we have to test (list)
    :param model: U-Net model (pytorch model)
    :param device: cpu or gpu device
    :param csv_no_filter: csv file to write the results
    :param it: run id (int)
    :param train_img: number of training img (when varying the number of training samples) (int)
    :param concat_res: csv file with aggregated results (mean over evry runs)
    :return: concat_res (+ evaluates the training step and write the results in csv files + save prediction imgs)
    """

    def predict_patch_side(preds_img, center, step, patch_size, inputs, target_patch):
        """
        :param preds_img: img with predictions (array)
        :param center: center of the 1rst patch to cut (list)
        :param step: overlapping coeff when sliding the patch (int)
        :param patch_size: size of the patch (int)
        :param inputs: tested image (array)
        :param target_patch: target patch (array)
        :return: border predictions of the large image
        """
        semi_patch = int(patch_size / 2)
        quart_patch = int(patch_size / 4)
        m, n, c = inputs.shape

        # slide the patch on the sides of the large image and make prediction on small patch
        while center[1] <= n - semi_patch and center[0] <= m - semi_patch:
            crop_img = inputs[center[0] - semi_patch:center[0] + semi_patch,
                       center[1] - semi_patch:center[1] + semi_patch, :]
            img, _ = test_transform(crop_img, target_patch)
            img = img.to(device, dtype=torch.float)
            img = torch.unsqueeze(img, 0)
            outputs = model(img)
            preds = outputs.argmax(1)
            predict = preds.cpu().numpy()
            predict = predict[0]
            # write the results in an array that has the shape of the large image
            if step[0] == 0:
                preds_img[center[0] - semi_patch:center[0] + semi_patch,
                center[1] - quart_patch:center[1] + quart_patch] = \
                    predict[:,quart_patch:quart_patch + semi_patch]
            if step[1] == 0:
                preds_img[center[0] - quart_patch:center[0] + quart_patch,
                center[1] - semi_patch:center[1] + semi_patch] = predict[quart_patch:quart_patch + semi_patch, :]
            center[0] = center[0] + step[0]
            center[1] = center[1] + step[1]
        return preds_img

    def predict_patch_inside(preds_img, center_init, step, patch_size, inputs, target_patch):
        """
        :param preds_img: img with predictions (array)
        :param center_init: center of the 1rst patch to cut (list)
        :param step: overlapping coeff when sliding the patch (int)
        :param patch_size: size of the patch (int)
        :param inputs: tested image (array)
        :param target_patch: target patch (array)
        :return: predictions of the large image (not on borders)
        """
        center = center_init.copy()
        semi_patch = int(patch_size / 2)
        quart_patch = int(patch_size / 4)
        m, n, c = inputs.shape
        while center[0] <= m - semi_patch:
            while center[1] <= n - semi_patch:
                crop_img = inputs[center[0] - semi_patch:center[0] + semi_patch,
                           center[1] - semi_patch:center[1] + semi_patch, :]
                img, _ = test_transform(crop_img, target_patch)

                img = img.to(device, dtype=torch.float)
                img = torch.unsqueeze(img, 0)
                outputs = model(img)
                preds = outputs.argmax(1)
                predict = preds.cpu().numpy()
                predict = predict[0]
                preds_img[center[0] - quart_patch:center[0] + quart_patch,
                center[1] - quart_patch:center[1] + quart_patch] = predict[quart_patch:quart_patch + semi_patch,
                                                                   quart_patch:quart_patch + semi_patch]
                center[1] = center[1] + step[1]
            center[0] = center[0] + step[0]
            center[1] = 64
        return preds_img

    save_dir = args.save_results + '/u_net_results/' + args.run_name + '/' + str(it) + "/"
    json_dir = args.save_results + '/s2ships/'
    save_json = args.save_results + '/u_net_results/' + args.run_name + '/'
    if not os.path.exists(save_dir):
        print('creating result directory...')
        os.makedirs(save_dir)
    model.eval()

    mean_EuroSAT = [0.4386203, 0.45689246, 0.45665017, 0.44572416, 0.47645673, 0.45139566]
    std_EuroSAT = [0.29738334, 0.29341888, 0.3096154, 0.28741345, 0.302901, 0.28648832]
    name_list = ['01_mask_rome', '02_mask_suez1', '03_mask_suez2', '04_mask_suez3', '05_mask_suez4', '06_mask_suez5',
                 '07_mask_suez6', '08_mask_brest1', '09_mask_panama', '10_mask_toulon', '11_mask_marseille',
                 '12_mask_portsmouth', '13_mask_rotterdam1', '14_mask_rotterdam2', '15_mask_rotterdam3',
                 '16_mask_southampton']

    # normalize test image according to training data normalization
    test_transform = T_seg.Compose([
        custom_transforms.ToTensor(),
        T_seg.Normalize(mean=mean_EuroSAT, std=std_EuroSAT),
    ])

    inputs, targets, img_id = dataset_test[0]

    # load and prepare annotation files
    with open(json_dir + 'coco-s2ships.json', 'r') as json_file:
        data = json_file.read()
    data_f = json.loads(data)
    coco_new = modify_coco_2_cats(data_f)
    with open(save_json + "targets_json.json", "wt") as file:
        file.write(json.dumps(coco_new))
    coco_new_one_cat = modify_coco(data_f)
    with open(save_json + "targets_json_one_cat.json", "wt") as file:
        file.write(json.dumps(coco_new_one_cat))

    if isinstance(test_ind, list):
        single_eval = False
    else:
        single_eval = True
        test_ind = [test_ind]

    # predict patch by patch (scanning the large tile) & transform prediction masks into coco annotations
    with torch.no_grad():
        for img_ind in test_ind:  # len(dataset_test)):
            print('test image : ', img_ind)
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
            print(np.max(inputs[0]))
            inputs, targets, img_id = dataset_test[img_ind - 1]
            m, n, c = inputs.shape
            preds_img = np.zeros((m, n))
            patch_size = 64
            # predict side
            # top side
            semi_patch = int(patch_size / 2)
            center = [semi_patch, semi_patch]
            step = [0, semi_patch]
            target_patch = targets[0:patch_size, 0:patch_size]
            preds_img = predict_patch_side(preds_img, center, step, patch_size, inputs, target_patch)

            # left side
            center = [semi_patch, semi_patch]
            step = [semi_patch, 0]
            preds_img = predict_patch_side(preds_img, center, step, patch_size, inputs, target_patch)

            # right side
            center = [semi_patch, n - semi_patch]
            step = [semi_patch, 0]
            preds_img = predict_patch_side(preds_img, center, step, patch_size, inputs, target_patch)

            # bottom side
            center = [m - semi_patch, semi_patch]
            step = [0, semi_patch]
            preds_img = predict_patch_side(preds_img, center, step, patch_size, inputs, target_patch)
            # predict inside
            center_init = [64, 64]
            step = [32, 32]
            preds_img = predict_patch_inside(preds_img, center_init, step, patch_size, inputs, target_patch)
            rgb_img = np.zeros((inputs.shape[0], inputs.shape[1], 3), dtype=np.uint8)

            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            rgb_img[:, :, 0] = clahe.apply(((inputs[:, :, 0]) * 255).astype(np.uint8))
            rgb_img[:, :, 1] = clahe.apply((inputs[:, :, 1] * 255).astype(np.uint8))
            rgb_img[:, :, 2] = clahe.apply((inputs[:, :, 2] * 255).astype(np.uint8))

            # for key in predictions_dict.keys():
            key = "no_filter"

            img = rgb_img

            idx = 0
            COLORS = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype='uint8')

            color = COLORS[0]

            if single_eval:
                # if leave-one-out setting
                np.save(save_dir + 'it{i}_{a}_predicted_segmentation_mask.npy'.format(i=it, a=name_list[
                    img_id - 1]), preds_img[:, :])
            else:
                # if vary the number of training samples, store by indicating how many training imgs were used
                np.save(save_dir + 'it{i}_{a}_predicted_segmentation_mask_train_img_{f}.npy'.format(i=it, a=name_list[
                    img_id - 1], f=train_img), preds_img[:, :])

            # segment the components
            preds_im = np.where(preds_img[:, :] == 1, 1, 0)
            preds_im = (preds_im * 255).astype(np.uint8)
            out_img = np.where(preds_im[..., None], color, img)
            for cat in range(2, args.num_classes):
                color = COLORS[cat - 1]
                preds_im = np.where(preds_img[:, :] == cat, 1, 0)
                preds_im = (preds_im * 255).astype(np.uint8)
                out_img = np.where(preds_im[..., None], color, out_img).astype(np.uint8)

            for cat in range(1, args.num_classes):
                color = COLORS[cat - 1]
                preds_im = np.where(preds_img[:, :] == cat, 1, 0)
                preds_im = (preds_im * 255).astype(np.uint8)
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

            if single_eval:
                cv.imwrite(save_dir + 'it{i}_{n}_visualization_{e}.png'.format(i=it, n=args.run_name,
                                                                               e=name_list[img_id - 1]), out_img)
            else:
                cv.imwrite(save_dir + 'it{i}_{n}_visualization_{e}_train_img_{f}.png'.format(i=it, n=args.run_name,
                                                                                             e=name_list[img_id - 1],
                                                                                             f=train_img), out_img)
            # prepare annotation files
            if len(pred_json["annotations"]) == 0:
                print('STOP', key)
            with open(save_json + "preds_json.json", "wt") as file:
                file.write(json.dumps(pred_json["annotations"], default=convert))
            cocoGt = COCO(save_json + "targets_json_one_cat.json")
            cocoDt = cocoGt.loadRes(save_json + "preds_json.json")
            cocoEval = COCOeval(cocoGt, cocoDt, "segm")

            per_size_tp_list = []
            per_size_nb_pos = []
            conf_list = []

            # all ship evaluation
            cocoEval.params.useCats = 0
            cocoEval.params.imgIds = img_id
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            conf_list.append(cocoEval.get_confusion_matrix())

            # per size evaluation
            tp_small, nb_positives_small = cocoEval.get_tp_pos_small()
            tp_large, nb_positives_large = cocoEval.get_tp_pos_large()
            per_size_tp_list.append(tp_small)
            per_size_tp_list.append(tp_large)
            per_size_nb_pos.append(nb_positives_small)
            per_size_nb_pos.append(nb_positives_large)

            # per class evaluation
            per_class_tp_list = []
            per_class_nb_pos = []
            cocoGt = COCO(save_json + "targets_json.json")
            for cat in range(1, 3):
                with open(save_json + 'preds_json.json', 'r') as json_file:
                    data = json_file.read()
                data_f = json.loads(data)
                new_coco = change_cat_id(data_f, cat)
                with open(save_json + "preds_json_modified.json", "wt") as file:
                    file.write(json.dumps(new_coco, default=convert))
                cocoDt = cocoGt.loadRes(save_json + "preds_json_modified.json")
                cocoEval = COCOeval(cocoGt, cocoDt, "segm")
                cocoEval.params.imgIds = img_id
                cocoEval.params.catIds = [cat]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                tp, nb_positives = cocoEval.get_tp_pos()
                per_class_tp_list.append(tp)
                per_class_nb_pos.append(nb_positives)

            # write results in csv file
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

            # prepare aggregated results
            precision = conf_mat[0] / (conf_mat[0] + conf_mat[1]) if (conf_mat[0] + conf_mat[1]) != 0 else 0
            recall = conf_mat[0] / (conf_mat[0] + conf_mat[2]) if (conf_mat[0] + conf_mat[2]) != 0 else 0
            F1 = 0
            if precision != 0 and recall != 0:
                F1 = 2 * precision * recall / (precision + recall)
            FA = conf_mat[1] / (17.80 * 9.30)
            sailing = per_class_tp_list[0] / per_class_nb_pos[0] if per_class_nb_pos[0] != 0 else 0
            moored = per_class_tp_list[1] / per_class_nb_pos[1] if per_class_nb_pos[1] != 0 else 0
            small = per_size_tp_list[0] / per_size_nb_pos[0] if per_size_nb_pos[0] != 0 else 0
            large = per_size_tp_list[1] / per_size_nb_pos[1] if per_size_nb_pos[1] != 0 else 0
            add_el = [precision, recall, F1, FA, sailing, moored, small, large]
            for j in range(concat_res.shape[1]):
                concat_res[img_id - 1][j] = concat_res[img_id - 1][j] + add_el[j]

    return concat_res


def load_data_test(test_dir):
    """generate the train and val dataloader, you can change this for your specific task
    Args:
        traindir (str): train dataset dir
        valdir (str): validation dataset dir
    Returns:
        tuple: the train dataset and validation dataset
    """
    dataset_test = s2ship_patch_test(test_dir, indices=[1, 2, 3, 7, 10, 11])
    return dataset_test


def load_data_train(traindir, excl_img_id, mi, ma):
    """generate the train and val dataloader, you can change this for your specific task
    Args:
        traindir (str): train dataset dir
        valdir (str): validation dataset dir
    Returns:
        tuple: the train dataset and validation dataset
    """
    mean_EuroSAT = [0.4386203, 0.45689246, 0.45665017, 0.44572416, 0.47645673, 0.45139566]
    std_EuroSAT = [0.29738334, 0.29341888, 0.3096154, 0.28741345, 0.302901, 0.28648832]

    train_transform = T_seg.Compose([
        # T_seg.Resize(args.input_size),
        T_seg.RandomHorizontalFlip(),
        T_seg.RandomVerticalFlip(),
        custom_transforms.ToTensor(),
        T_seg.Normalize(mean=mean_EuroSAT, std=std_EuroSAT),
    ])
    dataset = s2ship_patch(traindir, excl_img_id, mi, ma, transform=train_transform, indices=[1, 2, 3, 7, 10, 11])
    return dataset


def main_vary_img(args, it, concat_res):
    """
    :param args: parser arguments
    :param it: run id
    :param concat_res: csv file with aggregated results
    :return: one complete training run + varying the number of training samples
    """

    save_dir = args.save_results + '/u_net_results/' + args.run_name + '/' + str(it) + "/"
    if not os.path.exists(save_dir):
        print('creating result directory...')
        os.makedirs(save_dir)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    print('device : ', device)
    dict = vars(args)

    # prepare csv file
    first_row = ['Img Id', 'All TP', 'All FP', 'All FN', 'Sailing ships TP', 'Sailing ships total positives',
                 'Moored ships TP', 'Moored ships total positives', 'Small TP', 'Small total positives',
                 'Large TP', 'Large total positives']
    csv_no_filter = csv.writer(open(save_dir + '{c}_it{i}_confusion_matrix_no_filter.csv'.format(c=args.run_name, i=it),
                                    'wt'), lineterminator='\n', )
    print('run_name', args.run_name)
    list_img_train = [12, 15, 7, 11, 14, 16, 6, 9, 5, 1, 4, 10, 3]
    list_concat = [concat_res for i in range(len(list_img_train))]
    test_ind = [2, 8, 13]
    dataset_test = load_data_test(args.test_path)

    # get the 3rd and 97th percentile of the test dataset to clip the data values between 0 and 1
    mi, ma = dataset_test.get_min_max()
    if args.exp_ID is not None:
        # start mflow parent experiment
        mlflow.start_run(run_name='{n}_it{i}_PARENT_RUN'.format(n=args.run_name, i=it), experiment_id=args.exp_ID)

    # for each experiment with st number of training img
    for st in range(len(list_img_train)):
        concat_res = list_concat[st]

        # img to exclude from training (test img + some imgs in the training set)
        excl_img_id = list_img_train[st + 1:]
        excl_img_id.extend(test_ind)
        print('nb_train_img : ', len(list_img_train) - len(excl_img_id) + len(test_ind))
        csv_no_filter.writerow(['nb of train img', len(list_img_train) - len(excl_img_id) + len(test_ind)])

        if args.exp_ID is not None:
            mlflow.log_params(dict)
        dataset = load_data_train(args.train_path, excl_img_id, mi, ma)

        # count the number of training ships (not distinct training ships)
        nb_boats = 0
        for _, e in dataset:
            n_comp, labels, stats, centroids = cv.connectedComponentsWithStats((e.numpy() * 255).astype("uint8"))
            nb_boats += n_comp - 1
        print("nb boats dataset train (no distinct) :", int(nb_boats * args.val_split))
        csv_no_filter.writerow(['nb of ships (no distinct)', int(nb_boats * args.val_split)])
        csv_no_filter.writerow(first_row)

        # set dataloaders
        dataset_size = len(dataset)
        print('len dataset', dataset_size)
        indices = list(range(dataset_size))
        split1 = int(args.val_split * dataset_size)
        print('len training set : ', split1)

        np.random.shuffle(indices)

        train_indices, val_indices = indices[:split1], indices[split1:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=6, num_workers=args.num_workers, sampler=valid_sampler,
                                drop_last=True)

        # model
        resnet = resnet50(args.num_classes, in_channels=args.in_channels, pretrained=args.pretrained)

        # load SSL pretrained weights
        if args.weights:
            if 'pth' in args.weights:
                resnet_ssl = torch.load(args.weights)
                if args.dp:
                    resnet_ssl = resnet_ssl.module
                resnet = nn.Sequential(
                    nn.Conv2d(args.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                    *list(resnet_ssl.backbone.children())[1:],
                    nn.Linear(in_features=2048, out_features=args.num_classes))
            elif 'ckpt' in args.weights:
                checkpoint = torch.load(args.weights)
                model = Moco18_sat(input_size=args.input_size, channels=args.in_channels, num_ftrs=2048)
                model.load_state_dict(checkpoint['state_dict'])
                resnet = nn.Sequential(
                    nn.Conv2d(args.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                    *list(model.backbone.children())[1:],
                    nn.Linear(in_features=2048, out_features=args.num_classes))
            else:
                resnet = resnet50(args.num_classes, in_channels=args.in_channels, pretrained=args.pretrained)
                resnet.load_state_dict(torch.load(args.weights))

        model = UNet(resnet, n_classes=args.num_classes, mode=args.mode)
        model = model.float()

        model.to(device)
        if args.resume:
            model = torch.load(args.resume + '{}.pth'.format(test_ind + 1))
            # TODO: resume learning rate

        # loss
        if args.criterion == 'focal_loss':
            gamma = 2
            alphas = [.05]
            for i in range(1, args.num_classes):
                alphas.append(0.25)
            criterion = FocalLoss_b(gamma=gamma,
                                    alpha=torch.tensor(np.array(alphas), dtype=torch.float32, device=device))
        elif args.criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss().to(device)
        # criterion = nn.BCELoss().to(device)
        if args.exp_ID is not None:
            mlflow.log_param("Focal loss alphas", alphas)
            mlflow.log_param("Focal loss gamma", gamma)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        print('nb epochs', args.epochs)
        print('===============')
        print('START TRAINING')
        print('===============')
        loss_per_test = []
        jaccard_per_test = []

        if args.exp_ID is not None:
            # start mlflow child experiment
            mlflow.start_run(run_name=args.run_name + 'it{i}-nb_img_{n}'.format(n=st + 1, i=it),
                             experiment_id=args.exp_ID,
                             nested=True)
            mlflow.log_param("child", "yes")
            mlflow.log_param("nb_boats", int(nb_boats * args.val_split))

        # training steps
        for epoch in range(args.epochs):
            train_one_epoch(epoch, train_loader, model, criterion, optimizer, device)
            mean_loss, mean_jaccard = evalidation(epoch, val_loader, model, criterion, device)
            loss_per_test.append(mean_loss)
            jaccard_per_test.append(mean_jaccard)
            if args.exp_ID is not None:
                mlflow.log_metric("val loss", mean_loss, step=epoch)
                mlflow.log_metric("jaccard", mean_jaccard, step=epoch)
            lr_scheduler.step(mean_loss)
        if args.exp_ID is not None:
            # end mflow child experiment
            mlflow.end_run()

        print('===============')
        print('RUNNING TEST')
        print('===============')

        # evaluation on 3 test images
        concat_res = predictions(dataset_test, test_ind, model, device, csv_no_filter, it, st + 1, concat_res)
        list_concat[st] = concat_res

    if args.exp_ID is not None:
        # end mflow parent experiment
        mlflow.end_run()

    return list_concat


def main_all_img(args, it, concat_res):
    """
        :param args: parser arguments
        :param it: run id
        :param concat_res: csv file with aggregated results
        :return: one complete training run in a leave-one-out setting
        """
    save_dir = args.save_results + '/u_net_results/' + args.run_name + '/' + str(it) + "/"
    if not os.path.exists(save_dir):
        print('creating result directory...')
        os.makedirs(save_dir)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    print('device : ', device)

    list_img = np.arange(16)
    dict = vars(args)

    # initialize csv file
    first_row = ['Img Id', 'All TP', 'All FP', 'All FN', 'Sailing ships TP', 'Sailing ships total positives',
                 'Moored ships TP', 'Moored ships total positives', 'Small TP', 'Small total positives',
                 'Large TP', 'Large total positives']
    csv_no_filter = csv.writer(open(save_dir + '{c}_it{i}_confusion_matrix_no_filter.csv'.format(c=args.run_name, i=it),
                                    'wt'), lineterminator='\n', )
    csv_no_filter.writerow(first_row)
    print('run_name', args.run_name)
    dataset_test = load_data_test(args.test_path)

    # get normalization coeff
    mi, ma = dataset_test.get_min_max()

    if args.exp_ID is not None:
        # start mflow parent experiment
        mlflow.start_run(run_name='{n}_it{i}_PARENT_RUN'.format(n=args.run_name, i=it), experiment_id=args.exp_ID)
        mlflow.log_params(dict)

    # for each test img, train on the other images of S2-SHIPS dataset
    for test_ind in list_img:
        print("training step ", test_ind + 1)

        # dataset and dataloader
        nb_excl_img = 15 - args.nb_train_img
        if nb_excl_img > 0:
            a = np.arange(1, 17).tolist()
            a.pop(test_ind + 1)
            excl_img_id = random.sample(a, nb_excl_img)
            excl_img_id.append(test_ind)
            excl_img_id.sort()
        else:
            excl_img_id = [test_ind + 1]
        dataset = load_data_train(args.train_path, excl_img_id, mi, ma)
        dataset_size = len(dataset)
        print('len dataset', dataset_size)
        indices = list(range(dataset_size))
        split1 = int(args.val_split * dataset_size)
        print('len training set : ', split1)

        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split1], indices[split1:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=6, num_workers=args.num_workers, sampler=valid_sampler,
                                drop_last=True)
        # model
        resnet = resnet50(args.num_classes, in_channels=args.in_channels, pretrained=args.pretrained)
        # load SSL pretrained weights
        if args.weights:
            if 'pth' in args.weights:
                resnet_ssl = torch.load(args.weights)
                if args.dp:
                    resnet_ssl = resnet_ssl.module
                resnet = nn.Sequential(
                    nn.Conv2d(args.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                    *list(resnet_ssl.backbone.children())[1:],
                    nn.Linear(in_features=2048, out_features=args.num_classes))
            elif 'ckpt' in args.weights:
                checkpoint = torch.load(args.weights)
                model = Moco18_sat(input_size=args.input_size, channels=args.in_channels, num_ftrs=2048)
                model.load_state_dict(checkpoint['state_dict'])
                resnet = nn.Sequential(
                    nn.Conv2d(args.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                    *list(model.backbone.children())[1:],
                    nn.Linear(in_features=2048, out_features=args.num_classes))
            else:
                resnet = resnet50(args.num_classes, in_channels=args.in_channels, pretrained=args.pretrained)
                resnet.load_state_dict(torch.load(args.weights))

        model = UNet(resnet, n_classes=args.num_classes, mode=args.mode)
        model = model.float()

        model.to(device)
        if args.resume:
            model = torch.load(args.resume + '{}.pth'.format(test_ind + 1))

        # loss
        if args.criterion == 'focal_loss':
            gamma = 2
            alphas = [.05]
            for i in range(1, args.num_classes):
                alphas.append(0.25)
            criterion = FocalLoss_b(gamma=gamma,
                                    alpha=torch.tensor(np.array(alphas), dtype=torch.float32, device=device))
        elif args.criterion == 'cross_entropy':
            criterion = nn.CrossEntropyLoss().to(device)
        if args.exp_ID is not None:
            mlflow.log_param("Focal loss alphas", alphas)
            mlflow.log_param("Focal loss gamma", gamma)

        # optim and lr scheduler
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        print('nb epochs', args.epochs)
        print('===============')
        print('START TRAINING')
        print('===============')
        loss_per_test = []
        jaccard_per_test = []

        if args.exp_ID is not None:
            # start mlflow child experiment
            mlflow.start_run(run_name=args.run_name + 'it{i}-step_{n}'.format(n=test_ind + 1, i=it),
                             experiment_id=args.exp_ID, nested=True)
            mlflow.log_param("child", "yes_{n}".format(n=test_ind + 1))

        for epoch in range(args.epochs):
            train_one_epoch(epoch, train_loader, model, criterion, optimizer, device)
            mean_loss, mean_jaccard = evalidation(epoch, val_loader, model, criterion, device)
            loss_per_test.append(mean_loss)
            jaccard_per_test.append(mean_jaccard)
            if args.exp_ID is not None:
                mlflow.log_metric("val loss", mean_loss, step=epoch)
                mlflow.log_metric("jaccard", mean_jaccard, step=epoch)
            lr_scheduler.step(mean_loss)
        if args.exp_ID is not None:
            # end mlflow child experiment
            mlflow.end_run()

        print('===============')
        print('RUNNING TEST')
        print('===============')
        concat_res = predictions(dataset_test, test_ind + 1, model, device, csv_no_filter, it, 0, concat_res)
    if args.exp_ID is not None:
        # end mlflow parent experiment
        mlflow.end_run()
    return concat_res


def parse_args():
    parser = argparse.ArgumentParser(description='U-Net training on S2-SHIPS')
    parser.add_argument('--train-path', help='train dataset path')
    parser.add_argument('--test-path', help='validate dataset path')
    parser.add_argument('--weights', default=None, help='weights path')
    parser.add_argument('--criterion', default="focal_loss", help='')
    parser.add_argument('--pretrained', default=None, help='if use ResNet pretrained on ImageNet, give weights path')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--num-classes', default=3, type=int, help='num of classes')
    parser.add_argument('--in-channels', default=3, type=int, help='input image channels')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--input_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--save_results', default='./', help='path to save checkpoint')
    parser.add_argument('--exp_ID', default=None, type=int, help='mlflow exp ID')
    parser.add_argument('--nb_train_img', default=15, type=int, help='train/val split, must be less than 15')
    parser.add_argument('--val_split', default=0.9, type=float, help='train/val split, must be less than 1')
    parser.add_argument('--mode', default='tf', help='transfer learning(tf) or fine tunning(ft)')
    parser.add_argument('--iter', default=1, type=int, help='number of runs')
    parser.add_argument('--run_name', default='test', help='run name')
    parser.add_argument('--vary_nb_img', default=None, help='if vary nb of training samples, set something')
    parser.add_argument('--small_test', default=None, help='if test on only 2 imgs for debugging, set something')
    parser.add_argument('--dp', default=None, help='weights trained on multi gpu parallel')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    name_list = ['01_mask_rome', '02_mask_suez1', '03_mask_suez2', '04_mask_suez3', '05_mask_suez4', '06_mask_suez5',
                 '07_mask_suez6', '08_mask_brest1', '09_mask_panama', '10_mask_toulon', '11_mask_marseille',
                 '12_mask_portsmouth', '13_mask_rotterdam1', '14_mask_rotterdam2', '15_mask_rotterdam3',
                 '16_mask_southampton']
    args = parse_args()
    iterations = args.iter
    run_name = args.run_name
    torch.cuda.empty_cache()
    first_row_conc = ['Img Id', 'Prec', 'recall', 'F1', 'FA rate']

    # use mlflow backend if exp ID specified
    if args.exp_ID is not None:
        import mlflow

    if not os.path.exists(args.save_results + '/u_net_results/' + args.run_name + '/'):
        print('creating result directory...')
        os.makedirs(args.save_results + '/u_net_results/' + args.run_name + '/')

    # prepare csv aggregated results file
    concatenated_res = csv.writer(open(args.save_results + '/u_net_results/' + args.run_name + '/' +
                                       'concat_res_filter.csv', 'wt'), lineterminator='\n', )
    concatenated_res.writerow(first_row_conc)
    concat_res = np.zeros((16, 4))
    if args.vary_nb_img is not None:
        # run training for n runs
        for it in range(iterations):
            concat_res_list = main_vary_img(args, it, concat_res)
    else:
        # run training for n runs
        for it in range(iterations):
            concat_res = main_all_img(args, it, concat_res)
        for img_id in range(concat_res.shape[0]):
            for j in range(concat_res.shape[1]):
                concat_res[img_id][j] = concat_res[img_id][j] / iterations
            row = [name_list[img_id]]
            row.extend(list(concat_res[img_id]))

            concatenated_res.writerow(row)
