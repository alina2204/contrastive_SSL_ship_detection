from __future__ import print_function, division
import os
import tifffile
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from PIL import Image
import torch
from utils import get_band
from skimage.transform import resize

"""
EuroSat, AIS ship detection dataset Agenium Space, BigEarthNet (numpy version) and SEN12MS-A/RA dataloaders
"""


class EuroSAT12(Dataset):
    """EuroSAT 12 bands multispectral dataset."""

    def __init__(self,
                 input_dir: str, bands_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
                 transform: transforms.Compose = None):
        self.input_dir = input_dir

        # create a dictionary to store the path, filenames and associated targets for every samples
        keyDict = {"path", "filename", "label"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.target_list = []
        self.transform = transform
        dir_list = sorted(os.listdir(self.input_dir))
        for e in dir_list:
            if e not in self.target_list:
                self.target_list.append(e)
            sub_path = os.path.join(self.input_dir, e)
            subdir_list = sorted(os.listdir(sub_path))
            for h in subdir_list:
                self.dataset["path"].append(sub_path + '/' + h)
                self.dataset["filename"].append(e + '/' + h)
                self.dataset["label"].append(self.target_list.index(e))
        self.bands_idx = bands_idx

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        fname = self.dataset["filename"][index]
        img_dir = self.dataset["path"][index]
        target = self.dataset["label"][index]
        img = tifffile.imread(img_dir)

        # get image sample with the given multispectral indices
        idx = self.bands_idx
        h, w, c = img.shape
        sample = np.zeros((h, w, len(idx)), dtype=np.float32)
        index = 0
        for i in idx:
            channel = img[:, :, i]
            # clip and normalize the band
            mi, ma = np.nanpercentile(channel, (3, 97))
            norm_chan = np.clip((channel - mi) / (ma - mi + 0.0001), 0, 1)  # +0.0001 to avoid /0 error
            sample[:, :, index] = norm_chan
            index += 1

        if self.transform:
            sample = self.transform(sample)
        return sample, target, fname


class SEN12MS(Dataset):
    """SEN12MS 12 bands dataset, without cutting smaller patches.
    It is adapted for the RA pretext task since it has large region patches
    """

    def __init__(self,
                 input_dir: str, bands_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
                 transform: transforms.Compose = None):
        self.input_dir = input_dir
        # create a dictionary to store the path, filenames and associated targets for every samples
        keyDict = {"path", "filename", "label"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.target_list = []
        self.transform = transform
        dir_list = sorted(os.listdir(self.input_dir))
        for e in dir_list:
            sub_path = os.path.join(self.input_dir, e)
            subdir_list = sorted(os.listdir(sub_path))
            for h in subdir_list:
                subsub_path = os.path.join(sub_path, h)
                subsubdir_list = sorted(os.listdir(subsub_path))
                for el in subsubdir_list:
                    self.dataset["path"].append(subsub_path + '/' + el)
                    self.dataset["filename"].append(e + '/' + h + '/el')
                    self.dataset["label"].append(0)  # we don't care about the labels
        self.bands_idx = bands_idx

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        fname = self.dataset["filename"][index]
        img_dir = self.dataset["path"][index]
        target = self.dataset["label"][index]
        img = tifffile.imread(img_dir)
        idx = self.bands_idx
        # idx = list(np.arange(13))
        # idx.pop(10)
        # image = img[:,:,self.bands_idx]
        # sample = image
        h, w, c = img.shape
        sample = np.zeros((h, w, len(idx)), dtype=np.float32)
        index = 0
        for i in idx:
            channel = img[:, :, i]
            mi, ma = np.nanpercentile(channel, (3, 97))
            norm_chan = np.clip((channel - mi) / (ma - mi + 0.0001), 0, 1)
            # norm_chan = np.clip(channel / np.max(channel), 0, 1)
            sample[:, :, index] = norm_chan
            index += 1
        if self.transform:
            sample = self.transform(sample)
        return sample, target, fname


class SEN12MS_cut(Dataset):
    """SEN12MS 12 bands dataset, with smaller patches.
    It is adapted for the A (simple data augmentations) pretext task
    """

    def __init__(self,
                 input_dir: str, bands_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
                 transform: transforms.Compose = None):
        self.input_dir = input_dir
        # create a dictionary to store the path, filenames and associated targets for every samples
        keyDict = {"path", "filename", "label"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.target_list = []
        self.transform = transform
        dir_list = sorted(os.listdir(self.input_dir))
        for e in dir_list:
            sub_path = os.path.join(self.input_dir, e)
            subdir_list = sorted(os.listdir(sub_path))
            for h in subdir_list:
                subsub_path = os.path.join(sub_path, h)
                subsubdir_list = sorted(os.listdir(subsub_path))
                for el in subsubdir_list:
                    self.dataset["path"].append(subsub_path + '/' + el)
                    self.dataset["filename"].append(e + '/' + h + '/el')
                    self.dataset["label"].append(0)
        self.bands_idx = bands_idx
        self.dataset_size = len(self.dataset["filename"])
        self.len_data = 16 * self.dataset_size
        # within the large original patch of size 256*256, we can get 16 smaller patches of size 64*64.
        # The coordinates are the following:
        self.patch_coord = [[0, 64, 0, 64], [0, 64, 64, 128], [0, 64, 128, 192], [0, 64, 192, 256],
                            [64, 128, 0, 64], [64, 128, 64, 128], [64, 128, 128, 192], [64, 128, 192, 256],
                            [128, 192, 0, 64], [128, 192, 64, 128], [128, 192, 128, 192], [128, 192, 192, 256],
                            [192, 256, 0, 64], [192, 256, 64, 128], [192, 256, 128, 192], [192, 256, 192, 256]]

    def __len__(self):
        """Returns the length of the dataset.

        """
        return self.len_data

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        img_indx = index
        patch_id = -1
        while img_indx >= 0:
            patch_id += 1
            img_indx -= self.dataset_size
        img_indx += self.dataset_size
        patch_coo = self.patch_coord[patch_id]
        fname = self.dataset["filename"][img_indx]
        img_dir = self.dataset["path"][img_indx]
        target = self.dataset["label"][img_indx]
        img = tifffile.imread(img_dir)

        idx = self.bands_idx
        sample = np.zeros((64, 64, len(idx)), dtype=np.float32)
        index = 0
        for i in idx:
            channel = img[patch_coo[0]:patch_coo[1], patch_coo[2]:patch_coo[3], i]
            mi, ma = np.nanpercentile(channel, (3, 97))
            norm_chan = np.clip((channel - mi) / (ma - mi + 0.0001), 0, 1)
            sample[..., index] = norm_chan
            index += 1
        if self.transform:
            sample = self.transform(sample)
        return sample, target, fname


class BigEarthNet(Dataset):
    """BigEarthNet 12 bands dataset."""

    def __init__(self,
                 input_dir: str,
                 transform: transforms.Compose = None):
        self.input_dir = input_dir
        keyDict = {"path", "filename", "label"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.transform = transform
        dir_list = sorted(os.listdir(self.input_dir))
        for e in dir_list:
            sub_path = os.path.join(self.input_dir, e)
            self.dataset["path"].append(sub_path)
            self.dataset["filename"].append(e)
            self.dataset["label"].append(0)

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        Bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        fname = self.dataset["filename"][index]
        img_dir = self.dataset["path"][index]
        print(img_dir)
        target = self.dataset["label"][index]
        img_l = get_band(Bands, img_dir)
        h, w = img_l[2].shape
        sample = np.zeros((h, w, len(Bands)), dtype=np.float32)

        for band in range(len(img_l)):
            channel = img_l[band]
            channel = resize(channel, (h, w))
            mi, ma = np.nanpercentile(channel, (3, 97))
            norm_chan = np.clip((channel - mi) / (ma - mi + 0.00001), 0, 1)
            sample[:, :, band] = norm_chan
        if self.transform:
            sample = self.transform(sample)
        return sample, target, fname


class BigEarthNet_numpy(Dataset):
    """BigEarthNet 12 bands dataset from numpy arrays (in order to speed up the training process."""

    def __init__(self,
                 input_dir: str, idx=[1, 2, 3, 10, 11],
                 transform: transforms.Compose = None):
        self.input_dir = input_dir
        self.idx = idx
        keyDict = {"path", "filename", "label"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.transform = transform
        dir_list = sorted(os.listdir(self.input_dir))
        for e in dir_list:
            sub_path = os.path.join(self.input_dir, e)
            self.dataset["path"].append(sub_path)
            self.dataset["filename"].append(e)
            self.dataset["label"].append(0)
        self.good_ind = 0

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        fname = self.dataset["filename"][index]
        img_dir = self.dataset["path"][index]
        target = self.dataset["label"][index]
        sample = np.load(img_dir)
        idx = self.idx
        m, n, c = sample.shape
        bands5 = np.zeros((m, n, len(idx)))
        index = 0

        for i in idx:
            mi, ma = np.nanpercentile(sample[:, :, i], (3, 97))
            norm_chan = np.clip((sample[:, :, i] - mi) / (ma - mi + 0.00001), 0, 1)
            bands5[:, :, index] = norm_chan
            index += 1

        # if only nan values, debug by taking a working sample. Maybe this isn't necessary
        bands5 = np.nan_to_num(bands5, nan=0)
        if np.sum(bands5) == 0:
            fname = self.dataset["filename"][self.good_ind]
            img_dir = self.dataset["path"][self.good_ind]
            target = self.dataset["label"][self.good_ind]
            sample = np.load(img_dir)
            idx = self.idx
            m, n, c = sample.shape
            bands5 = np.zeros((m, n, len(idx)))
            index = 0

            for i in idx:
                mi, ma = np.nanpercentile(sample[:, :, i], (3, 97))
                norm_chan = np.clip((sample[:, :, i] - mi) / (ma - mi + 0.00001), 0, 1)
                bands5[:, :, index] = norm_chan
                index += 1
        else:
            self.good_ind = index

        if self.transform:
            bands5 = self.transform(bands5)
        return bands5, target, fname


class AIS_sentinel2(Dataset):
    """AIS Sentinel 2 RGB bands dataset.
    A csv file is used to indicate which sample to take for pretext or downstream task.
    """

    def __init__(self,
                 data_dir: str, csv_dir: str,
                 transform: transforms.Compose = None):
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        keyDict = {"path", "filename", "label"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.transform = transform
        with open(self.csv_dir) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ImageId'] != '':
                    self.dataset["path"].append(self.data_dir + '/' + row['ImageId'])
                    self.dataset["filename"].append(row['ImageId'])
                    self.dataset["label"].append(int(row['labels']))

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        fname = self.dataset["filename"][index]
        sample = Image.open(self.dataset["path"][index])
        target = self.dataset["label"][index]

        if self.transform:
            sample = self.transform(sample)
        return sample, target, fname


class AIS_sentinel2_5(Dataset):
    """AIS Sentinel 2 multispectral 5 bands dataset.
     A csv file is used to indicate which sample to take for pretext or downstream task.
     """

    def __init__(self,
                 data_dir: str, csv_dir: str,
                 transform: transforms.Compose = None):
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        keyDict = {"path", "filename", "label"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.transform = transform
        with open(self.csv_dir) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ImageId'] != '':
                    self.dataset["path"].append(self.data_dir + '/' + row['ImageId'])
                    self.dataset["filename"].append(row['ImageId'])
                    self.dataset["label"].append(int(row['labels']))

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        fname = self.dataset["filename"][index]
        sample = tifffile.imread(self.dataset["path"][index]).astype(np.float32)
        _, _, c = sample.shape
        for i in range(c):
            sample[:, :, i] = sample[:, :, i] / 255.
        target = self.dataset["label"][index]

        if self.transform:
            sample = self.transform(sample)
        return sample, target, fname


class s2ship_patch(Dataset):
    """s2ship 12 bands training dataset (patches).
    some special args here :
    - excl_img_id : imgs ids to exclude (like test image, or in case wanting to train on a smaller amount of data)
    - min_data : 3rd percentile calculated over S2-SHIPS dataset
    - min_data : 97th percentile calculated over S2-SHIPS dataset
    """

    def __init__(self,
                 input_dir: str,
                 excl_img_id, min_data, max_data,
                 transform: transforms.Compose = None, indices=[1, 2, 3, 10, 11]):
        name_list = ['rome', 'suez1', 'suez2', 'suez3', 'suez4',
                     'suez5',
                     'suez6', 'brest1', 'panama', 'toulon', 'marseille',
                     'portsmouth', 'rotterdam1', 'rotterdam2', 'rotterdam3',
                     'southampton']  # name of original tiles
        self.input_dir = input_dir
        self.indices = indices
        keyDict = {"filename"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.transform = transform

        dir_list = sorted(os.listdir(self.input_dir))

        # exclude images if needed (e.g. test image)
        for e in dir_list:
            if excl_img_id is not None:
                flag = 0
                for el in excl_img_id:
                    if name_list[el - 1] in e:
                        flag = 1
            # if this patch does not require to be excluded, add it to the list
            if flag == 0:
                self.dataset["filename"].append(e)
        self.mi = min_data
        self.ma = max_data

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """

        fname = self.dataset["filename"][index]
        img_and_label = np.load(self.input_dir + '/' + fname, allow_pickle=True)
        sample = img_and_label.item().get("data")
        m, n, c = sample.shape
        bands5 = np.zeros((m, n, len(self.indices)))
        index = 0
        for i in self.indices:
            # mi, ma = np.nanpercentile(sample[:,:,i], (3, 97))  # if calculated on each patch
            # normalisation
            norm_chan = np.clip((sample[:, :, i] - self.mi[i]) / (self.ma[i] - self.mi[i] + 0.000001), 0, 1)
            bands5[:, :, index] = norm_chan
            index += 1
        target = img_and_label.item().get("label")
        if self.transform:
            bands5, target = self.transform(bands5, target)
            target = torch.squeeze(target)
        return bands5, target


class s2ship_patch_test(Dataset):
    """s2ship 12 bands test dataset (full tiles).
    """

    def __init__(self,
                 input_dir: str,
                 transform: transforms.Compose = None, indices=[1, 2, 3, 10, 11]):
        self.input_dir = input_dir
        self.indices = indices

        keyDict = {"filename"}
        self.dataset = dict([(key, []) for key in keyDict])
        self.transform = transform
        # calculate the 3rd and 97th percentile for further image processing
        min_data = 999999 * np.ones((12, 1))
        max_data = -999999 * np.ones((12, 1))
        dir_list = sorted(os.listdir(self.input_dir))
        for e in dir_list:
            self.dataset["filename"].append(e)
            img_and_label = np.load(self.input_dir + '/' + e, allow_pickle=True)
            sample = img_and_label.item().get("data")
            for i in self.indices:
                mi, ma = np.nanpercentile(sample[:, :, i], (3, 97))
                if mi < min_data[i]:
                    min_data[i] = mi
                if ma > max_data[i]:
                    max_data[i] = ma
        self.mi = min_data
        self.ma = max_data

    def get_min_max(self):
        """Returns 3rd and 97th percentiles of the dataset for further image processing.

                """
        return self.mi, self.ma

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset["filename"])

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """

        fname = self.dataset["filename"][index]
        index_img = fname[:2]

        img_and_label = np.load(self.input_dir + '/' + fname, allow_pickle=True)
        sample = img_and_label.item().get("data")
        m, n, c = sample.shape
        bands5 = np.zeros((m, n, len(self.indices)))
        index = 0
        for i in self.indices:
            # mi, ma = np.nanpercentile(sample[:, :, i], (3, 97))  # if patch per patch normalisation
            norm_chan = np.clip((sample[:, :, i] - self.mi[i]) / (self.ma[i] - self.mi[i] + 0.000001), 0, 1)
            bands5[:, :, index] = norm_chan
            index += 1
        target = img_and_label.item().get("label")
        if self.transform:
            bands5, target = self.transform(bands5, target)
            target = torch.squeeze(target)
        return bands5, target, int(index_img)
