import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from torchvision import transforms

from transforms import *


class PascalVOCDataset(Dataset):
    """
    Pascal VOC Dataset 2012/2017 for object detection
    """
    def __int__(self, data_folder: str, split: str, keep_difficult: bool = False):
        """
        :param data_folder: folder where data is kept
        :param split: 'TRAIN' or 'TEST'
        :param keep_difficult: keep difficult ground truth objects or discard them
        """
        self.split = split.upper()
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        #  Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Get objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        resize = Resize((300, 300))
        to_tensor = ToTensor()
        normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        random_crop = RandomCrop(300)
        random_horizontal_flip = RandomHorizontalFlip(0.5)
        photo_dist = PhotometricDistort()
        composed = transforms.Compose([resize, random_crop, random_horizontal_flip, photo_dist, to_tensor, normalize])
        image, boxes, labels, difficulties = composed(image, boxes, labels, difficulties)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images, boxes, labels, difficulties = [], [], [], []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)  # (N, 3, 300, 300), 3 lists of N tensors each

        return images, boxes, labels, difficulties
