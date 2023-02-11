import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from dataset import PascalVOCDataset
from model import *
from tqdm import tqdm

torch.cuda.empty_cache()

# Data parameters
data_folder = r'D:\ObjectDetection\PascalVOC'  # folder with data files
keep_difficult = True  # difficult objects to detect

n_classes = len(label_map)  # number of different types of objects

# Learning parameters
checkpoint: str = 'checkpoints/checkpoint_ssd300.pt'  # path to model checkpoint, None if none
batch_size: int = 25  # batch size
iterations: int = 120000  # number of iterations to train
workers: int = 0  # number of workers for loading data in the DataLoader
print_freq: int = 100  # print training status every __ batches
lr: float = 1e-3  # learning rate
decay_lr_at: [int] = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to: float = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum: float = 0.9  # momentum
weight_decay: float = 5e-4  # weight decay
grad_clip: float = 0.0  # clip if gradients are exploding

cudnn.benchmark = True


def main():
    """
    Training.
    :return: checkpoint
    """
    global checkpoint, decay_lr_at

    if checkpoint is None:
        start_epoch = 0
        # Initialize model or load checkpoint
        print("Initializing model...")
        model = SSD300(n_classes=n_classes)
        biases, not_biases = [], []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, collate_fn=train_dataset.collate_fn)

    # Epochs
    epochs = iterations // (len(train_dataset) // 32)  # the paper trains for 120k iteration with a batch size of 32
    print('Training for %d epochs...' % epochs)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for i, (images, boxes, labels, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not 0.0:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time: Value = {batch_time.val:.3f} (Average = {batch_time.avg:.3f})\t'
                  'Data Time: Value = {data_time.val:.3f} (Average = {data_time.avg:.3f})\t'
                  'Loss = {loss.val:.4f}'.format(epoch, i, len(train_loader),
                                                 batch_time=batch_time,
                                                 data_time=data_time,
                                                 loss=losses))

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
