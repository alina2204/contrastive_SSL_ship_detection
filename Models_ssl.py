import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly
import torchvision

""" This script contains all the SSL models used for SSL pretraining :
- MoCo RGB - ResNet 18
- SimSiam RGB - ResNet 18
- Barlow Twins RGB - ResNet 18
- MoCo multispectral - ResNet 50
 
 see https://github.com/lightly-ai/lightly for more tutorials/infos """


class Moco18(pl.LightningModule):
    # MoCo RGB model
    def __init__(self, memory_bank_size=4096, max_epochs=100, num_ftrs=512, pretrained=False, temperature=0.1,
                 momentum=0.9, lr=6e-2, weight_decay=5e-4):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # create a moco based on ResNet
        self.resnet_SSL = \
            lightly.models.MoCo(self.backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(temperature=temperature, memory_bank_size=memory_bank_size)
        self.max_epochs = max_epochs
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        self.resnet_SSL(x)

    # log weights in tensorboard
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_SSL(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_SSL.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class SimSiam18(pl.LightningModule):
    # SiamSiam RGB model
    def __init__(self, max_epochs=100, num_ftrs=512, pretrained=False, momentum=0.9, lr=6e-2, weight_decay=5e-4,
                 input_size=64, channels=3):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(channels, input_size, kernel_size=7)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # create a moco based on ResNet
        self.resnet_SSL = \
            lightly.models.SimSiam(self.backbone, num_ftrs=num_ftrs, proj_hidden_dim=512,
                                   pred_hidden_dim=128, out_dim=512, num_mlp_layers=2)

        # create our loss
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        self.max_epochs = max_epochs
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        self.resnet_SSL(x)

    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_SSL(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_SSL.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class BarlowTwins18(pl.LightningModule):
    # BarlowTwins RGB model
    def __init__(self, max_epochs=100, num_ftrs=512, pretrained=False, momentum=0.9, lr=6e-2, weight_decay=5e-4,
                 input_size=64, channels=3):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(channels, input_size, kernel_size=7)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # create a moco based on ResNet
        self.resnet_SSL = \
            lightly.models.BarlowTwins(self.backbone, num_ftrs=num_ftrs, proj_hidden_dim=512,
                                       out_dim=512, num_mlp_layers=2)

        # create our loss
        self.criterion = lightly.loss.barlow_twins_loss.BarlowTwinsLoss()
        self.max_epochs = max_epochs
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        self.resnet_SSL(x)

    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_SSL(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_SSL.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class Moco18_sat(pl.LightningModule):
    # MoCo multispectral model
    def __init__(self, memory_bank_size=4096, max_epochs=100, num_ftrs=512, pretrained=False, temperature=0.1,
                 momentum=0.9, lr=1e-3, weight_decay=5e-4, input_size=64, channels=12):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(channels, input_size, kernel_size=7)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # create a moco based on ResNet
        self.resnet_SSL = \
            lightly.models.MoCo(self.backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(temperature=temperature, memory_bank_size=memory_bank_size)
        self.max_epochs = max_epochs
        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):

        self.resnet_SSL(x)

    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        self.resnet_SSL.train()
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_SSL(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_SSL.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]
