import pytorch_lightning as pl
from torchsat.models.classification import resnet18, resnet50
from utils import *

"""
Classifier based on ResNet, ResNet and U-Net for segmentation based on ResNet encoder
"""
class Classifier(pl.LightningModule):
    """ simple ResNet classifier using a ResNet backbone (downstream task classification)
    agrs :
    - model : ResNet weights
    - nb_class : number of class to be classified
    - mode : tf for transfer learning (freezing backbone layers), ft for finetuning
    """
    def __init__(self, model, max_epochs=100, nb_class=10, mode='tf'):
        super().__init__()
        # upload a SSL pretrained model based on ResNet
        self.resnet = model
        # freeze the layers of the SSL algo if tf (transfer learning) mode
        if mode == 'tf':
            for p in self.resnet.parameters():  # reset requires_grad
                p.requires_grad = False

        # add the classification head
        self.fc = nn.Linear(2048, nb_class)
        self.max_epochs = max_epochs
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet.backbone(x)
            y_hat = y_hat.squeeze()  # supposes batch_size>1
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def predict(self, x):  # for batch_size=1
        with torch.no_grad():
            y_hat = self.resnet.backbone(x)

            y_hat = y_hat.squeeze(2)
            y_hat = y_hat.squeeze(2)
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.fc.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class ResNet18(pl.LightningModule):
    """ simple ResNet classifier using a ResNet backbone (downstream task classification)
    for ImageNet or training from scratch
        agrs :
        - model : ResNet weights
        - nb_class : number of class to be classified
        """
    def __init__(self,  max_epochs=100, pretrained=True, nb_class=10, in_channel=3):
        super().__init__()
        # create a ResNet50 or 18
        if in_channel > 3:
            self.resnet = resnet50(nb_class, in_channels=in_channel, pretrained=pretrained)
            # self.resnet = resnet18(nb_class, in_channels=in_channel, pretrained=pretrained)
        else:
            self.resnet = resnet50(3, pretrained=pretrained)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, nb_class)
        self.max_epochs = max_epochs
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        y_hat = self.resnet(x)
        return y_hat

    def predict(self, x):  # for batch_size=1
        with torch.no_grad():
            y_hat = self.resnet(x)
        return y_hat

    def custom_histogram_weights(self):
        pass
        # for name, params in self.named_parameters():
        #     self.logger.experiment.add_histogram(
        #         name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):

        x, y, _ = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]


class UNet(nn.Module):
    """ UNet classifier using a ResNet backbone (downstream task classification)
            agrs :
            - model : ResNet weights
            - nb_class : number of class to be classified
            - mode : tf for transfer learning (freezing backbone layers), ft for finetuning
    implementation based on this github project : https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder
            """
    DEPTH = 6  # depth of the UNet

    def __init__(self, resnet, n_classes=2, mode='tf'):
        super().__init__()
        down_blocks = []
        up_blocks = []

        # freeze the weights if transfer learning
        if mode == 'tf':
            for child in resnet.children():
                for p in child.parameters():
                    p.requires_grad = False

        # initialize the unet based on resnet backbone
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 6, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
