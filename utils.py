import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
voc_labels = ('background',  # always index 0
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')

label_map = {k: v for v, k in enumerate(voc_labels)}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map
voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

label_colormap = dict(zip(label_map.keys(), voc_colormap))


def parse_annotation(annotation_path):
    """
    Parse an annotation given its path
    :param annotation_path:
    :return: dict containing lists of bounding boxes, labels, difficulties
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []
    difficulties = []

    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')

        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the labels and bounding boxes.
    :param voc07_path:
    :param voc12_path:
    :param output_folder:
    :return:
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = []
    train_objects = []
    n_objects = 0

    # training data
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in the training set
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))

            if len(objects['boxes']) == 0:
                continue

            # Add annotation's objects to list
            train_objects.append(objects)
            n_objects += len(objects)

            # Add image to list
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_images) == len(train_objects)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print(f'Number of training images: {len(train_images)}')
    print(f'Number of training objects: {n_objects}')
    print(f'Path to output folder: {output_folder}')

    # validation data
    test_images = []
    test_objects = []
    n_objects = 0

    # Find IDs of images in the training set
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))

        if len(objects['boxes']) == 0:
            continue

        # Add annotation's objects to list
        test_objects.append(objects)
        n_objects += len(objects)

        # Add image to list
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_images) == len(test_objects)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print(f'Number of test images: {len(test_images)}')
    print(f'Number of test objects: {n_objects}')
    print(f'Path to output folder: {output_folder}')


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. down-sample by keeping every m-th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


if __name__ == '__main__':

    VOC07_PATH = r'D:\ObjectDetection\PascalVOC\2007\VOCdevkit\VOC2007'
    VOC12_PATH = r'D:\ObjectDetection\PascalVOC\2012\VOCdevkit\VOC2012'
    OUTPUT_FOLDER = r'D:\ObjectDetection\PascalVOC'

    create_data_lists(voc07_path=VOC07_PATH,
                      voc12_path=VOC12_PATH,
                      output_folder=OUTPUT_FOLDER)