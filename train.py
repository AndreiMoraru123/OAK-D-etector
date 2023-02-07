import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils import *
from dataset import PascalVOCDataset

# Data parameters
data_folder = r'D:\ObjectDetection\PascalVOC'  # folder with data files
keep_difficult = True  # difficult objects to detect

n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 0  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding

cudnn.benchmark = True

train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=keep_difficult)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=workers, collate_fn=train_dataset.collate_fn)

#  testing the train loader
for i, (images, boxes, labels, _) in enumerate(train_loader):
    print(images.shape)
    print(len(boxes))
    print(len(labels))
    break
