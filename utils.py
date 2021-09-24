import tifffile
import os
from typing import Optional
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import cv2 as cv

"""Some utility functions/class : 
- get_band : select a given band from a directory containing an image spectral 
bands into subdirectories 
- modify_coco, change_cat_id, convert... : scripts to change COCO annotations file
- ConvBlock, Bridge, UpBlockForUNetWithResNet50 : blocks for U-Net taken from 
https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder 
- Focal_loss_b : focal loss for the U-Net taken from 
https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py """


def get_band(L, folder):
    """
    Code adapted from : https://github.com/alina2204/plastic_detection_from_space

    Parameters
    ----------
    L : string list pointing to the spectral bands to use.
        example : L = ['B08','BO4','True_color']
        'True_color' is the RGB image
    folder : String pointing to the name of the folder to search. Each folder
        have the spectral bands of an image for a given date and place
        exemple : '2019_04_18_M'
    Returns
    -------
    band_dict : dictionnary of np.array(), 1 array pour 1 spectral band
        where coeff are  normalizaed reflectivies for the considered spectral
        bands
        exemple : band_dict = {'B08' : array, 'B04' : 'array'}
    """

    # Test of the parameters
    if len(L) == 0 or type(L) != list:
        raise TypeError("L is not a list, or is empty")
    if type(folder) != str:
        raise TypeError("folder is not a string")

    # get paths to load spectral bands
    path = folder
    inside_folder = sorted(os.listdir(path))

    band_dict = []

    # load wanted spectral bands
    for raw_band in inside_folder:
        for desired_band in L:
            if desired_band in raw_band:
                band = tifffile.imread(os.path.join(path, raw_band))
                band_dict.append(band)

    # add the name of the folder to the dict
    # band_dict['folder'] = folder

    return band_dict

def modify_coco(coco):
    """

    :param coco: json file containing coco ground truth, loaded with coco
    :return: json file containing coco ground truth where each object (all sailing ships) is segmented separately
    """
    anns = coco['annotations']
    L_im = [[] for i in range(16)]
    idx = 0
    for i in range(len(anns)):
        if anns[i]["id"] > idx:
            idx = anns[i]["id"]
        length = len(anns[i]["segmentation"])
        img_id = anns[i]["image_id"]
        coco['annotations'][i]["category_id"] = 1
        coco['annotations'][i]["iscrowd"] = 0
        # if an ID has more than one object, save these objects and assign them a new ID
        if length > 1:
            for k in range(1, length):
                L_im[img_id - 1].append(anns[i]["segmentation"][k])
        coco['annotations'][i]["segmentation"] = [anns[i]["segmentation"][0]]
        mat_contour = np.asarray(anns[i]["segmentation"][0], dtype="float32")
        mat_contour = mat_contour.reshape((-1, 2))
        mask_area = cv.contourArea(mat_contour)
        coco['annotations'][i]["area"] = mask_area
    # add new objects ID
    idx += 1
    for j in range(16):
        for e in L_im[j]:
            new_annot = {
                "id": idx,
                "image_id": j + 1,
                "category_id": 1,
                "segmentation": [e],
                "iscrowd": 0
            }
            mat_contour = np.asarray(e, dtype="float32")
            mat_contour = mat_contour.reshape((-1, 2))
            mask_area = cv.contourArea(mat_contour)
            new_annot["area"] = mask_area
            coco["annotations"].append(new_annot)
            idx += 1
    return coco


def change_cat_id(coco_ann, id):
    """

    :param coco_ann: json file containing coco ground truth

    :param id: id that will replace the former category id
    :return: json file containing coco ground truth where each object id is replaced by the id given in argument
    """
    idx = 0
    for i in range(len(coco_ann)):
        if coco_ann[i]["id"] > idx:
            idx = coco_ann[i]["id"]
        coco_ann[i]["category_id"] = id
    return coco_ann


def modify_coco_2_cats(coco):
    """
        :param coco: json file containing coco ground truth with 2 categories (moored ships and sailing ships)
        :return: json file containing coco ground truth where each object (sailing and moored ships)
        is segmented separately
        """
    anns = coco['annotations']
    L_im = [[] for i in range(16)]
    cat_id = [[] for i in range(16)]
    idx = 0
    for i in range(len(anns)):
        if anns[i]["id"] > idx:
            idx = anns[i]["id"]
        length = len(anns[i]["segmentation"])
        img_id = anns[i]["image_id"]
        coco['annotations'][i]["category_id"] = 2 if anns[i]["category_id"] == 4 else 1
        coco['annotations'][i]["iscrowd"] = 0
        if length > 1:
            for k in range(1, length):
                L_im[img_id - 1].append(anns[i]["segmentation"][k])
                cat_id[img_id - 1].append(coco['annotations'][i]["category_id"])
        coco['annotations'][i]["segmentation"] = [anns[i]["segmentation"][0]]
        mat_contour = np.asarray(anns[i]["segmentation"][0], dtype="float32")
        mat_contour = mat_contour.reshape((-1, 2))
        mask_area = cv.contourArea(mat_contour)
        coco['annotations'][i]["area"] = mask_area
    idx += 1
    for j in range(16):
        for t in range(len(L_im[j])):
            new_annot = {
                "id": idx,
                "image_id": j + 1,
                "category_id": cat_id[j][t],
                "segmentation": [L_im[j][t]],
                "iscrowd": 0
            }
            mat_contour = np.asarray(L_im[j][t], dtype="float32")
            mat_contour = mat_contour.reshape((-1, 2))
            mask_area = cv.contourArea(mat_contour)
            new_annot["area"] = mask_area
            coco["annotations"].append(new_annot)
            idx += 1
    return coco


def convert(o):
    if isinstance(o, np.int32):
        return int(o)
    if isinstance(o, np.int64):
        return int(o)
    print('wrong type : ', type(o))
    raise TypeError


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class FocalLoss_b(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
