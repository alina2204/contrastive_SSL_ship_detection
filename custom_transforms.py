from torchsat.transforms import transforms_cls, functional
import torchvision
import numpy as np
import random
import torch
from PIL import Image


class RandomRotation(object):
    """Randomly rotate an image with a probability p.

    """

    def __init__(self, p=0.5):
        """ Args:
                    p (float): probability of rotation.
                """
        self.prob = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: rotated image.
        """
        angle_list = [-90, 90, 180]
        prob = np.random.random_sample()
        if prob < self.prob:
            return torchvision.transforms.functional.rotate(img, random.choice(angle_list))
        return img

    def __repr__(self):
        pass


class RandomGaussianBlur(object):
    """Apply gaussian blur with a probability p
    """

    def __init__(self, p=0.5):
        """ Args:
            p (float): probability.
                        """
        self.prob = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred.
        Returns:
            PIL Image: blurred image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            transform = transforms_cls.GaussianBlur(kernel_size=3)
            return transform(img)
        return img

    def __repr__(self):
        pass


class RandomResizedCrop(object):
    """Crop and resize image with a probability p
    """

    def __init__(self, p=0.5):
        """ Args:
            p (float): probability.
        """
        self.prob = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be resized and cropped.
        Returns:
            PIL Image: crop + resized image.
        """
        prob = np.random.random_sample()
        input_size = img.shape[0]
        if prob < self.prob:
            transform = transforms_cls.RandomResizedCrop(int((1 - prob / 10) * input_size), input_size,
                                                         interpolation=Image.BICUBIC)
            return transform(img)
        return img

    def __repr__(self):
        pass


class RandomColorJitter(object):
    """Apply random color jitter with a probability p
    """

    def __init__(self, p=0.5, bright=0.1, contrast=0.1):
        """ Args:
                    p (float): probability.
                    bright (float) : brightness interval
                    contrast (float): contrast interval
                """
        self.prob = p
        self.bright = bright * np.random.random_sample()
        self.contrast = contrast * np.random.random_sample()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: transformed image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            factor = random.uniform(0.04, self.contrast)  # min value should be high enough in order to avoid black img
            img = functional.adjust_contrast(img, factor)
        prob = np.random.random_sample()
        if prob < self.prob:
            transform = transforms_cls.RandomBrightness(max_value=self.bright)
            img = transform(img)
        return img

    def __repr__(self):
        pass


class ToTensor(object):
    def __call__(self, img, mask):
        return functional.to_tensor(img), torch.tensor(mask, dtype=torch.long)
