import torch.nn.functional as F
import lightly
import pytorch_lightning as pl
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
import os
# mlflow
import mlflow.pytorch

# utils
from Models_ssl import Moco18, SimSiam18, BarlowTwins18, Moco18_sat
from Classifiers import Classifier, ResNet18
from datasets import EuroSAT12, BigEarthNet, AIS_sentinel2, AIS_sentinel2_5
from custom_transforms import *
import csv

""" ------------------------ 
    This script can be use for classification training and testing.
    This can be for either the SSL downstream task or simple ResNet baseline
    ------------------------ """

""" ------------------------ 
    Pretext task - parameters 
    ------------------------ """

# args parser
parser = argparse.ArgumentParser(description='SSL - downstream task training')
parser.add_argument('--dataset', help='downstream task dataset path')
parser.add_argument('--run_name', help='description of experiment (dataset, pretext task)')
parser.add_argument('--channels', type=int, default=12, help='number of channels (RGB = 3, multispectral can be >3)')
parser.add_argument('--weights', default='./pretrained_moco_e2_b256.tar', help='ssl weights')
parser.add_argument('--model', default='moco', help='moco_sat, imagenet, scratch, simsiam, moco or barlow twins.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--split', type=float, default=0.9, help='val/test split')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--input_size', type=int, default=64, help='size of training patches')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--exp_ID', type=int, default=None, help='if using mlflow, give experiment ID')
parser.add_argument('--save_dir', default='./')
parser.add_argument('--nb_class', type=int, default=2)
parser.add_argument('--test', default=None, type = float, help='if test step set the test split, must be less than 1')
parser.add_argument('--csv_train', default=None, help='csv file with training images path for AIS ships')
parser.add_argument('--csv_test', default=None, help='csv file with test images path for AIS ships')
parser.add_argument('--mode', default='tf')

args = parser.parse_args()

# Settings
workers = args.workers
epochs = args.epochs
input_size = args.input_size

# set the path to the downstream task dataset
path_to_data = args.dataset

# save directory
save_dir = args.save_dir + '/downstream_task/' + args.run_name + '/'
if not os.path.exists(save_dir):
    print('creating result directory...')
    os.makedirs(save_dir)
csv_no_filter = csv.writer(open(save_dir + '{}_confusion_matrix_classification.csv'.format(args.run_name),
                                'wt'), lineterminator='\n', )

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

if args.channels == 12:
    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
elif args.channels == 5:
    bands = [1, 2, 3, 10, 11]
elif args.channels == 6:
    bands = [1, 2, 3, 7, 10, 11]

# Augmentations
if args.model != 'moco_sat' and args.channels <= 3:
    train_classifier_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=mean_EuroSAT[1:4],
            std=std_EuroSAT[1:4],
        )
    ])
elif input_size < 64:
    train_classifier_transforms = transforms_cls.Compose([
        torchvision.transforms.ToTensor(),
        transforms_cls.Normalize(mean=[mean_EuroSAT[index] for index in bands],
                                 std=[std_EuroSAT[index] for index in bands]),
    ])
else:
    train_classifier_transforms = transforms_cls.Compose([
        transforms_cls.RandomCrop(input_size),
        torchvision.transforms.ToTensor(),
        transforms_cls.Normalize(mean=[mean_EuroSAT[index] for index in bands],
                                 std=[std_EuroSAT[index] for index in bands]),
    ])

# get dataloaders
if 'EuroSAT' in path_to_data:
    if args.channels == 12:
        dataset_train_classifier = EuroSAT12(input_dir=path_to_data, transform=train_classifier_transforms)
        dataset_test_classifier = EuroSAT12(input_dir=path_to_data, transform=train_classifier_transforms)
    if args.channels == 5:
        dataset_train_classifier = EuroSAT12(input_dir=path_to_data, transform=train_classifier_transforms,
                                             bands_idx=[1, 2, 3, 10, 11])
        dataset_test_classifier = EuroSAT12(input_dir=path_to_data, bands_idx=[1, 2, 3, 10, 11],
                                            transform=train_classifier_transforms)

elif 'BigEarthNet' in path_to_data:
    dataset_train_classifier = BigEarthNet(input_dir=path_to_data, transform=train_classifier_transforms)
    dataset_test_classifier = BigEarthNet(input_dir=path_to_data, transform=train_classifier_transforms)
elif args.csv_train and 'AIS' in args.csv_train:
    if args.channels == 5:
        dataset_train_classifier = AIS_sentinel2_5(data_dir=path_to_data, csv_dir=args.csv_train,
                                                   transform=train_classifier_transforms)
        dataset_test_classifier = AIS_sentinel2_5(data_dir=path_to_data, csv_dir=args.csv_test,
                                                  transform=train_classifier_transforms)
        bands = [1, 2, 3, 10, 11]
    else:
        dataset_train_classifier = AIS_sentinel2(data_dir=path_to_data, csv_dir=args.csv_train,
                                                 transform=train_classifier_transforms)
        dataset_test_classifier = AIS_sentinel2(data_dir=path_to_data, csv_dir=args.csv_test,
                                                transform=train_classifier_transforms)
else:
    dataset_train_classifier = lightly.data.LightlyDataset(input_dir=path_to_data,
                                                           transform=train_classifier_transforms)
    dataset_test_classifier = lightly.data.LightlyDataset(input_dir=path_to_data, transform=train_classifier_transforms)

# train/val random split
dataset_size = len(dataset_train_classifier)
indices = list(range(dataset_size))
split1 = int(np.floor(args.split * dataset_size))
batch_size = 10 if split1 < args.batch_size else args.batch_size
print('len training set : ', split1)
csv_no_filter.writerow(['len training', split1])
csv_no_filter.writerow(['TN', 'FP', 'FN', 'TP'])
np.random.shuffle(indices)

# if test step, create 3 datasets
if args.test:
    if args.test <= args.split:
        args.test = args.split + (1 - args.split) / 2
    split2 = int(np.floor(args.test * dataset_size))
    print('len validation set', split2 - split1)
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]

# else, create training and val sets only
else:
    train_indices, val_indices = indices[:split1], indices[split1:]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    drop_last=True,
    num_workers=workers,
    sampler=train_sampler
)

dataloader_val_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=workers,
    sampler=valid_sampler
)

""" ------------------------ 
    Downstream task - training 
    ------------------------ """

# create the classifiers based on pretrained SSL backbone
if args.model == 'moco':
    # create the MoCo model
    model = Moco18(max_epochs=epochs, num_ftrs=512)
    # load the weights trained on pretext task
    if 'pth' in args.weights:
        model = torch.load(args.weights)
    else:
        model.load_state_dict(torch.load(args.weights))
    model.eval()
    # create the classifier
    classifier = Classifier(model.resnet_SSL, nb_class=args.nb_class, mode=args.mode)

elif args.model == 'simsiam':
    # create the SimSiam model
    model = SimSiam18(max_epochs=epochs)
    # load the weights trained on pretext task
    if 'pth' in args.weights:
        model = torch.load(args.weights)
    else:
        model.load_state_dict(torch.load(args.weights))
    model.eval()
    # create the classifier
    classifier = Classifier(model.resnet_SSL, nb_class=args.nb_class, mode=args.mode)

elif args.model == 'barlow':
    # create the SimSiam model
    model = BarlowTwins18(max_epochs=epochs)
    # load the weights trained on pretext task
    if 'pth' in args.weights:
        model = torch.load(args.weights)
    else:
        model.load_state_dict(torch.load(args.weights))
    model.eval()
    # create the classifier
    classifier = Classifier(model.resnet_SSL, nb_class=args.nb_class, mode=args.mode)

elif args.model == 'moco_sat':
    # create the SimSiam model
    model = Moco18_sat(max_epochs=epochs, channels=args.channels, num_ftrs=2048)
    # load the weights trained on pretext task
    if 'pth' in args.weights:
        model = torch.load(args.weights)
    else:
        model.load_state_dict(torch.load(args.weights))
    # create the classifier
    classifier = Classifier(model.resnet_SSL, nb_class=args.nb_class, mode=args.mode)

elif args.model == 'scratch':
    # ResNet18 model trained from scratch
    classifier = ResNet18(nb_class=args.nb_class, pretrained=False, in_channel=args.channels)
elif args.model == 'imagenet':
    # ResNet18 model pretrained on ImageNet
    classifier = ResNet18(nb_class=args.nb_class, pretrained=True, in_channel=args.channels)
else:
    print('Model not implemented, use MoCo18 instead...')
    model = Moco18(max_epochs=epochs)
    # load the weights trained on pretext task
    if 'pth' in args.weights:
        model = torch.load(args.weights)
    else:
        model.load_state_dict(torch.load(args.weights))
    model.eval()
    # create the classifier
    classifier = Classifier(model.resnet_SSL, nb_class=args.nb_class, mode=args.mode)

classifier.train()
#  mlflow logs
if args.exp_ID:
    mlflow.pytorch.autolog()

gpus = args.gpus if torch.cuda.is_available() else 0

# Pytorch lightning trainer
if gpus >= 2:
    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, distributed_backend='ddp', progress_bar_refresh_rate=20)
else:
    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, progress_bar_refresh_rate=20)
# fit the trainer
if args.exp_ID:
    with mlflow.start_run(experiment_id=args.exp_ID) as run:  # run mlflow
        trainer.fit(classifier, dataloader_train_classifier, dataloader_val_classifier)
else:
    trainer.fit(classifier, dataloader_train_classifier, dataloader_val_classifier)

""" ------------------------ 
    Downstream task - test 
    ------------------------ """
if args.test:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier.to(device)
    predictions = []
    labels = []
    probs = []
    print('len test set', len(dataset_test_classifier))
    classifier.eval()
    # Do the predictions
    for i in test_indices:
        x_val, y_val, _ = dataset_test_classifier[i]
        x_val = x_val.to(device)
        x_val = x_val.unsqueeze(0)
        y_hat = classifier.predict(x_val)
        _, preds = torch.max(y_hat, 1)
        labels.append(y_val)
        predictions.append(preds.item())
        preds = np.squeeze(preds.cpu().numpy())
        with torch.no_grad():
            prob = F.softmax(y_hat.cpu(), dim=1).numpy()
        prob = prob[0][preds]
        probs.append(prob)

    # print confusion matrix
    cm = confusion_matrix(labels, predictions)
    print('Confusion Matrix : \n', cm)
    # print classification report
    print(classification_report(labels, predictions))
    tn, fp, fn, tp = cm.ravel()

    # save confusion matrix
    csv_no_filter.writerow([tn, fp, fn, tp])
