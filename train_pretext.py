import torch
import torchvision
import numpy as np
import lightly
import pytorch_lightning as pl
import argparse
import os
from datasets import EuroSAT12, AIS_sentinel2, BigEarthNet_numpy, AIS_sentinel2_5, SEN12MS, SEN12MS_cut
from torchsat.transforms import transforms_cls
from custom_transforms import RandomRotation, RandomGaussianBlur, RandomColorJitter, RandomResizedCrop
# for clustering and 2d representations
from sklearn import random_projection
from torch.utils.data.sampler import SubsetRandomSampler
# mlflow
import mlflow.pytorch
# utils
from Models_ssl import Moco18, SimSiam18, BarlowTwins18, Moco18_sat
from pretext_plots import create_filenames_embeddings, get_scatter_plot_with_thumbnails, plot_nearest_neighbors_3x3

""" ------------------------ 
    SSL pretext task pretraining
    ------------------------ """

""" ------------------------ 
    Pretext task - parameters 
    ------------------------ """
# args parser
parser = argparse.ArgumentParser(description='SSL - pretext task training')
parser.add_argument('--dataset', help='pretext task dataset path')
parser.add_argument('--run_name', help='description of experiment (dataset, pretext task)')
parser.add_argument('--csv', default=None, help='csv file with pretext task images path for AIS ships')
parser.add_argument('--weights', default=None)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--input_size', type=int, default=64, help='size of training patches')
parser.add_argument('--model', default='moco', help='moco_sat, simsiam, moco or barlow twins.')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--exp_ID', type=int, default=None, help='if using mlflow, give experiment ID')
parser.add_argument('--save_dir', default='./')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--channels', type=int, default=12, help='number of channels (RGB = 3, multispectral can be >3)')
parser.add_argument('--val_split', type=float, default=1., help='validation split')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--RA', type=int, default=None, help='if region based pretext task, indicate region size')

args = parser.parse_args()

# environment variables, disable visualization if running in a remote server
ENABLE_VIS = os.environ.get('DIS_VIS')

# SSL settings
num_workers = args.workers
batch_size = args.batch_size
epochs = args.epochs
input_size = args.input_size

# seed torch and numpy if deterministic experiments
seed = 1
torch.manual_seed(0)
np.random.seed(0)

# set the path to the pretext task dataset
path_to_data = args.dataset  # ./EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/'
torch.cuda.empty_cache()

save_dir = args.save_dir + '/pretext_task_pretraining/'+args.run_name+'/'
if not os.path.exists(save_dir):
    print('creating result directory...')
    os.makedirs(save_dir)

""" ------------------------ 
    Data loaders and augmentations setup 
    ------------------------
    
Setup data augmentations and loaders
"""

# mean and standard dev calulated on EuroSAT dataset
mean_EuroSAT = [0.44929576, 0.4386203, 0.45689246, 0.45665017, 0.47687784, 0.44870496,
                0.44587377, 0.44572416, 0.4612574, 0.3974199, 0.47645673, 0.45139566]
std_EuroSAT = [0.2883096, 0.29738334, 0.29341888, 0.3096154, 0.29744068, 0.28400135,
               0.2871275, 0.28741345, 0.27953532, 0.22587752, 0.302901, 0.28648832]

# if RGB
bands = [1, 2, 3]

# get sen12ms dataloaders for pretext task A or RA
if 'SEN12MS' in path_to_data and args.model == 'moco_sat':
    if args.channels == 12:
        dataset_train = SEN12MS(input_dir=path_to_data)
        bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if args.channels == 6:
        bands = [1, 2, 3, 7, 11, 12]
        mean_EuroSAT = [0.44929576, 0.4386203, 0.45689246, 0.45665017, 0.47687784, 0.44870496,
                        0.44587377, 0.44572416, 0.4612574, 0.3974199, 0.46645673, 0.47645673, 0.45139566]
        std_EuroSAT = [0.2883096, 0.29738334, 0.29341888, 0.3096154, 0.29744068, 0.28400135,
                       0.2871275, 0.28741345, 0.27953532, 0.22587752, 0.292901, 0.302901, 0.28648832]
        # if only data augmentation approach, cut large patch into several small patches
        if args.RA:
            # pretext  task RA
            dataset_train = SEN12MS(input_dir=path_to_data, bands_idx=bands)
        else:
            # pretext task A
            dataset_train = SEN12MS_cut(input_dir=path_to_data, bands_idx=bands)

# get EuroSAT dataloaders for pretext task A
elif 'EuroSAT' in path_to_data and args.model == 'moco_sat':
    if args.channels == 12:
        dataset_train = EuroSAT12(input_dir=path_to_data)
        bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if args.channels == 6:
        bands = [1, 2, 3, 7, 10, 11]
        dataset_train = EuroSAT12(input_dir=path_to_data, bands_idx=bands)

# get BEN dataloaders for pretext task A
elif 'BigEarthNet' in path_to_data:
    if args.channels == 5:
        bands = [1, 2, 3, 10, 11]
        dataset_train = BigEarthNet_numpy(input_dir=path_to_data, idx=bands)
    elif args.channels == 6:
        bands = [1, 2, 3, 7, 10, 11]
        dataset_train = BigEarthNet_numpy(input_dir=path_to_data, idx=bands)
    else:
        dataset_train = BigEarthNet_numpy(input_dir=path_to_data)

# get AIS ships (Agenium space) dataloaders for pretext task A
elif args.csv and 'AIS' in args.csv:
    if args.model == 'moco_sat':
        dataset_train = AIS_sentinel2_5(data_dir=path_to_data, csv_dir=args.csv)
        bands = [1, 2, 3, 10, 11]
    else:
        dataset_train = AIS_sentinel2(data_dir=path_to_data, csv_dir=args.csv, transform=torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(input_size)]))
else:
    dataset_train = lightly.data.LightlyDataset(input_dir=path_to_data)

print("length training set : ", len(dataset_train))

# create a collate function. Data augmentation chosen similar to MoCo v2 model

if args.model != 'moco_sat':
    # collate function for RGB images
    collate_fn = \
        lightly.data.ImageCollateFunction(input_size=input_size, hf_prob=0.5,
                                          vf_prob=0.5,  # require invariance to flips and rotations
                                          rr_prob=0.5,  # satellite images are all taken from the same height
                                          min_scale=0.5,  # so we use only slight random cropping
                                          cj_prob=0.3,  # weak color jitter for invariance w.r.t small color changes
                                          cj_bright=0.1,
                                          cj_contrast=0.1,
                                          cj_hue=0.1,
                                          cj_sat=0.1,
                                          normalize={'mean': mean_EuroSAT[1:4], 'std': std_EuroSAT[1:4]})
elif input_size < 64:
    collate_data_aug = transforms_cls.Compose([
        RandomResizedCrop(p=0.7),
        transforms_cls.RandomHorizontalFlip(p=0.5),
        transforms_cls.RandomVerticalFlip(p=0.5),
        RandomColorJitter(p=0.3, bright=0.1, contrast=0.1),
        RandomGaussianBlur(p=0.5),
        torchvision.transforms.ToTensor(),
        transforms_cls.Normalize(mean=[mean_EuroSAT[index] for index in bands],
                                 std=[std_EuroSAT[index] for index in bands]),
        RandomRotation(p=0.5)
    ])

    collate_fn = lightly.data.BaseCollateFunction(collate_data_aug)
else:
    if args.RA:
        augmentations = [
            transforms_cls.CenterCrop(args.RA),  # get the central region of each patch
            transforms_cls.RandomCrop(input_size),  # if pretext task region (random patch in a given region)
            RandomResizedCrop(p=0.7),
            transforms_cls.RandomHorizontalFlip(p=0.5),
            transforms_cls.RandomVerticalFlip(p=0.5),
            RandomColorJitter(p=0.3, bright=0.1, contrast=0.2),
            RandomGaussianBlur(p=0.4),
            torchvision.transforms.ToTensor(),
            transforms_cls.Normalize(mean=[mean_EuroSAT[index] for index in bands],
                                     std=[std_EuroSAT[index] for index in bands]),
            RandomRotation(p=0.7),
            torchvision.transforms.ConvertImageDtype(torch.float)
        ]
    else:
        augmentations = [
            RandomResizedCrop(p=0.7),
            transforms_cls.RandomHorizontalFlip(p=0.5),
            transforms_cls.RandomVerticalFlip(p=0.5),
            RandomColorJitter(p=0.3, bright=0.1, contrast=0.2),
            RandomGaussianBlur(p=0.4),
            torchvision.transforms.ToTensor(),
            transforms_cls.Normalize(mean=[mean_EuroSAT[index] for index in bands],
                                     std=[std_EuroSAT[index] for index in bands]),
            RandomRotation(p=0.7),
            torchvision.transforms.ConvertImageDtype(torch.float)
        ]
    collate_data_aug = transforms_cls.Compose(augmentations)

    collate_fn = lightly.data.BaseCollateFunction(collate_data_aug)

# training/test split
indices = list(range(len(dataset_train)))
split1 = int(args.val_split * len(dataset_train))
print('length training set : ', split1)
np.random.shuffle(indices)

train_indices, val_indices = indices[:split1], indices[split1:]
train_sampler = SubsetRandomSampler(train_indices)

# dataloader
dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers)

# if RGB images :  visualization of the classification of features in 2D space after doing the pretext task

if args.model != 'moco_sat':
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=mean_EuroSAT[1:4],
            std=std_EuroSAT[1:4],
        )
    ])

    # create a lightly dataset for embedding
    dataset_test = lightly.data.LightlyDataset(input_dir=path_to_data, transform=test_transforms)

    # create a dataloader for embedding
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

""" ------------------------ 
    Pretext task - training 
    ------------------------ """

if args.exp_ID:
    # mlflow logs
    mlflow.pytorch.autolog(log_models=True)

gpus = args.gpus if torch.cuda.is_available() else 0

if args.model == 'moco':
    # create the MoCo model
    model = Moco18(max_epochs=epochs, num_ftrs=2048)
elif args.model == 'moco_sat':
    # create the MoCo multispectral model
    model = Moco18_sat(max_epochs=epochs, input_size=input_size, channels=args.channels, num_ftrs=2048, lr=args.lr)
elif args.model == 'simsiam':
    # create the SimSiam model
    model = SimSiam18(max_epochs=epochs, input_size=input_size)
elif args.model == 'barlow':
    # create the BarlowTwins model
    model = BarlowTwins18(max_epochs=epochs, input_size=input_size)
else:
    print('Model not implemented, use MoCo18 instead...')
    model = Moco18(max_epochs=epochs)

if args.weights:
    print('loading pretrained weights...')
    model.load_state_dict(torch.load(args.weights))
# print(model)

# Pytorch lightning trainer
if gpus >= 2:
    print('parallel backend')
    trainer = pl.Trainer(max_epochs=epochs, gpus=-1, progress_bar_refresh_rate=20, accelerator='dp',
                         deterministic=False, precision=32, default_root_dir=save_dir, check_val_every_n_epoch=10)
else:
    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, progress_bar_refresh_rate=20, deterministic=False, precision=32,
                         default_root_dir=save_dir, check_val_every_n_epoch=10)

# fit the trainer

# if mlflow backend
if args.exp_ID:
    with mlflow.start_run(run_name=args.run_name, experiment_id=args.exp_ID) as run:
        trainer.fit(model, dataloader_train)
else:
    trainer.fit(model, dataloader_train)

# save the model's weights
torch.save(model, save_dir + args.run_name + '_{model}_e{e}_b{b}_channels{c}.pth'.format(model=args.model, e=epochs,
                                                                                         b=batch_size, c=args.channels))

""" ------------------------
    Pretext task - visualizations
    ------------------------ """

# if RGB images, show KNN and scatter plots
if args.model != 'moco_sat':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    filenames, embeddings = create_filenames_embeddings(model, dataloader_test)

    # ## Scatter Plot and Nearest Neighbors
    # for the scatter plot we want to transform the images to a two-dimensional
    # vector space using a random Gaussian projection
    projection = random_projection.GaussianRandomProjection(n_components=2)
    embeddings_2d = projection.fit_transform(embeddings)

    # normalize the embeddings to fit in the [0, 1] square
    M = np.max(embeddings_2d, axis=0)
    m = np.min(embeddings_2d, axis=0)
    embeddings_2d = (embeddings_2d - m) / (M - m)

    # get a scatter plot with thumbnail overlays
    get_scatter_plot_with_thumbnails(embeddings_2d, path_to_data, filenames, ENABLE_VIS, save_dir)

    # Nearest neighbors
    # try to plot KNN for some classes
    example_images = [
        'AnnualCrop',  # annual crop
        'Residential',  # buildings
        'River',  # river
        'SeaLake',  # lake
    ]

    # show example images for each cluster
    for i, example_image in enumerate(example_images):
        plot_nearest_neighbors_3x3(example_image, i, path_to_data, filenames, embeddings, ENABLE_VIS, save_dir)
