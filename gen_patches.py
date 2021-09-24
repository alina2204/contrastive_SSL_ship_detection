import numpy as np
import utils
import argparse
import os

"""
Generate small labeled patches from S2-SHIPS dataset, or generate the whole annotated dataset, given the labels
"""

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', help='train dataset path')
parser.add_argument('--gen', help='if generating small patches of size 64*64 enter patch')

args = parser.parse_args()
target_names = ['08_mask_brest1', '11_mask_marseille', '09_mask_panama', '12_mask_portsmouth', '01_mask_rome',
                '13_mask_rotterdam1', '14_mask_rotterdam2', '15_mask_rotterdam3', '16_mask_southampton',
                '02_mask_suez1', '03_mask_suez2', '04_mask_suez3', '05_mask_suez4', '06_mask_suez5', '07_mask_suez6',
                '10_mask_toulon']
list_names = ['brest1', 'marseille', 'panama', 'portsmouth', 'rome',
              'rotterdam1', 'rotterdam2', 'rotterdam3', 'southampton',
              'suez1', 'suez2', 'suez3', 'suez4', 'suez5', 'suez6',
              'toulon']
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
count_boats = 0
count_non_boats = 0

if args.gen == 'patch':
    # generate small patches
    print('generating small patches with labels')
    save_dir_path = args.save_dir + '/s2ships/patches_64_boats_only/'
    if not os.path.exists(save_dir_path):
        print('creating result directory...')
        os.makedirs(save_dir_path)
    for i in range(len(list_names)):
        name = list_names[i]
        target_name = target_names[i]
        z = 0
        img_list = utils.get_band(bands, args.save_dir + '/s2ships/' + name)
        target = np.load(args.save_dir + '/s2ships/s2ships_labels_npy/' + target_name + '_rgb.png.npy')
        m, n = img_list[0].shape

        patch_size = 64
        final_img = np.zeros((patch_size, patch_size, 12))  # last channel for labels
        print(target.shape[-1])
        final_label = np.zeros((patch_size, patch_size, target.shape[-1]))
        pix = [0, 0]
        step = 32
        while pix[0] <= m - patch_size:
            while pix[1] <= n - patch_size:
                i = 0
                for image in img_list:
                    c_img = image[pix[0]:pix[0] + patch_size, pix[1]:pix[1] + patch_size]
                    final_img[:, :, i] = c_img
                    i += 1
                final_label = target[pix[0]:pix[0] + patch_size, pix[1]:pix[1] + patch_size, :]

                if np.count_nonzero(final_label[:, :, 0] == 1) > 5:  # if there is at least 5 boat pixel,
                    # we keep the patch for the training set and save the img and the label(s) in a dictionary
                    if target.shape[-1] == 2:
                        save_dict = {"data": final_img, "label": final_label[:, :, 0] + 2 * final_label[:, :, 1]}
                    if target.shape[-1] == 1:
                        save_dict = {"data": final_img, "label": final_label}
                    np.save(save_dir_path + name + '_boat_patch_' + str(z) + '.npy', save_dict)
                z += 1
                pix[1] += step
            pix[0] += step
            pix[1] = 0


else:
    # dataset (full img+labels into dictionary), full size image
    print('generating full sized dataset with labels')
    save_dir_path = args.save_dir + '/s2ships/dataset_full/'
    if not os.path.exists(save_dir_path):
        print('creating result directory...')
        os.makedirs(save_dir_path)
    for i in range(len(list_names)):
        name = list_names[i]
        target_name = target_names[i]
        img_list = utils.get_band(bands, args.save_dir + '/s2ships/' + name)
        target = np.load(args.save_dir + '/s2ships/s2ships_labels_npy/' + target_name + '_rgb.png.npy')
        print('id', int(target_name[:2]))
        m, n = img_list[0].shape

        final_img = np.zeros((m, n, 12))  # last channel for labels
        i = 0
        for image in img_list:
            final_img[:, :, i] = image
            i += 1

        if target.shape[-1] == 2:
            save_dict = {"data": final_img, "label": target[:, :, 0] + 2 * target[:, :, 1]}
        if target.shape[-1] == 1:
            save_dict = {"data": final_img, "label": target}
        np.save(save_dir_path + target_name + '.npy', save_dict)
