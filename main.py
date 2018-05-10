import model
import model_bn
import torch.utils.data.dataloader as dl
import ISBI2012Data as ISBI
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import numpy as np
from PIL import Image
import random

import time
import shutil
import argparse


# saves the model with learning rate and weight decay
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_lr_' + str(args.lr) + "_wd_" + str(args.weight_decay) + '.pth.tar')


def train(trainloader, model, criterion, optimizer, epoch):

    model.train()
    loss_sum = 0
    for i, data in enumerate(trainloader):

        # get train and label data
        train, label = data
        # put on gpu or cpu
        train = train.to(device)
        # label is of type TensorLong
        label = label.to(device)

        # for the CrossEntropyLoss:
        # outputs needs to be of size: Minibatch, classsize, dim 1, dim 2 , ...
        # outputs  are 2 classes with 2d images. channelsize = class size
        # label needs to be of format: Minibatch, dim 1, dim 2, ...
        # I cut the channel info for it to work, because it is only a 2d image.
        # As an alternative, one could add 1 channel for class in train, than label does not need any change
        # label normally looks like: ( minibatchsize, 1, width, height )
        label = label.view(label.size(0), label.size(2), label.size(3))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # if padding is true, one row and one column is added at left, right, top and bottom at each convolution
        # to maintain the original size of the image
        outputs = model(train, padding=args.pad)
        # the log is needed to calculate the crossentropy
        loss = criterion(torch.log(outputs), label)
        loss.backward()

        optimizer.step()

        running_loss = loss.item()
        loss_sum = loss_sum + running_loss

        # save the first minibatch image in each loop
        # to save every image,, just remove the & (i == 0) part
        if args.save_images & (i == 0):
            save_images(outputs, './val/' + str(args.lr) + '/', epoch, index=i)

        # delete all references to variables:
        # https://discuss.pytorch.org/t/tensor-to-vari
        del outputs, train, label, loss
    loss_avg = loss_sum / (i + 1)

    return loss_avg


def eval(valloader, model, criterion, save_image):

    # switch the model to eval mode ( important for dropout layers or batchnorm layers )
    model.eval()
    loss_sum = 0
    for i, data in enumerate(valloader):
        # get train and label data
        val, label = data
        # put on gpu or cpu
        val = val.to(device)
        # label is of type TensorLong
        label = label.to(device)

        # for the CrossEntropyLoss:
        # outputs needs to be of size: Minibatch, classsize, dim 1, dim 2 , ...
        # outputs  are 2 classes with 2d images. Channelsize = class size
        # label needs to be of format: Minibatch, dim 1, dim 2, ...
        # i cut the channel info for it to work, because it is only a 2d image.
        # as an alternative, one could add 1 channel for class in train, than label does not need any change
        # label normally looks like: ( minibatchsize, 1, width, height )
        label = label.view(label.size(0), label.size(2), label.size(3))

        # forward + backward + optimize
        outputs = model(val, padding=args.pad)
        loss = criterion(torch.log(outputs), label)
        running_loss = loss.item()
        loss_sum = loss_sum + running_loss

        if save_image:
            save_image(outputs, './eval/', 'eval', index=i)

        del outputs, val, label, loss

    loss_avg = loss_sum / (i + 1)
    return loss_avg


def save_images(outputs, directory, epoch, index):
    # copy first image in outputs back to cpu and save it
    x = outputs[0][0][:][:].cpu().detach().numpy()
    y = outputs[0][1][:][:].cpu().detach().numpy()

    # convert image to save it properly
    x = (x * 255).astype(np.uint8)
    y = (y * 255).astype(np.uint8)
    x = Image.fromarray(x)
    if not os.path.exists(directory):
        os.makedirs(directory)
    x.save(directory + 'class1_' + str(epoch) + '_image_' + str(index) + '.jpg')
    y = Image.fromarray(y)
    y.save(directory + 'class2_' + str(epoch) + '_image_' + str(index) + '.jpg')


# Parameters can be set at command-line

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='dataset', choices=['ISBI2012', 'CTC2015'],
                    help='ISBI2012 or CTC2015')
# parser.add_argument(-'batch-size', type=int,
#                     metavar='N', help='training image batch size')
parser.add_argument('-mbs', '--mini-batch-size', dest='minibatchsize', type=int,
                    metavar='N', default=1, help='mini batch size (default: 1). '
                                                 'For 8k memory on gpu, minibatchsize of 2-3 possible')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('-lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum (default: 0.99)')
parser.add_argument('-es', '--epochsave', default=1, type=int, metavar='M',
                    help='save model every M epoch (default: 1)')
parser.add_argument('--weight-decay', '-wd', default=0, type=float,
                    metavar='W', help='weight decay (L2 penalty ) (default:0')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='relative path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-s', '--save-images', action='store_true',
                    help='save the first image of output each epoche')
parser.add_argument('-c', '--cpu', action ='store_true',
                    help='use cpu instead of gpu')
parser.add_argument('-p', '--pad', action ='store_true',
                    help='use padding at each 3x3 convolution to maintain image size')
parser.add_argument('-txt', action ='store_true',
                    help='save console output in txt')
parser.add_argument('-bn', action ='store_true',
                    help='use u-net with batchnorm layers added after each convolution')


args = parser.parse_args()
args.start_epoch = 0
best_loss = 10

# use same seed for testing purpose
torch.manual_seed(999)
random.seed(999)

print "***** Starting Programm *****"


# 1: design model

# check if cuda is available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.cpu:
    device = torch.device("cpu")

# .to(device) sends the data to the given device ( cuda or cpu )
if args.bn:
    model = model_bn.Unet().to(device)
else:
    model = model.Unet().to(device)

# use cudnn for better speed, if available
if device.type == "cuda":
    cudnn.benchmark = True

# 2: Construct loss and optimizer

# Using a softmax layer at the end, applying the log and using NLLoss()
# has the same loss as using no softmax layer, and calculating the CrossEntropyLoss()
# the difference is in the output image of the model.
# If you want to use the CrossEntropyLoss(), remove the softmax layer, and  the torch.log() at the loss

# criterion = nn.CrossEntropyLoss().to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Reduce learning rate when a metric has stopped improving, needs to be activated in epoch too
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

# load the ISBI 2012 training data
# the CTC2015 Datasetloader is not finished yet
# The length of the Dataset has to be set by yourself
# self, gloob_dir_train, gloob_dir_label, length, is_pad, eval, totensor):
if args.data == "ISBI2012":
    trainset = ISBI.ISBIDataset(
        "./ISBI 2012/Train-Volume/train-volume-*.tif", "./ISBI 2012/Train-Labels/train-labels-*.tif",
        length=24, is_pad=args.pad, eval=False, totensor=True)

    valset = ISBI.ISBIDataset(
        "./ISBI 2012/Val-Volume/train-volume-*.tif", "./ISBI 2012/Val-Labels/train-labels-*.tif",
        length=6, is_pad=args.pad, eval=True, totensor=True)
elif args.data == "CTC2015":
    trainset = ISBI.ISBIDataset(
        "./ISBI 2012/Train-Volume/train-volume-*.tif", "./ISBI 2012/Train-Labels/train-labels-*.tif",
        length=24, is_pad=args.pad, eval=False, totensor=True)

    valset = ISBI.ISBIDataset(
        "./ISBI 2012/Val-Volume/train-volume-*.tif", "./ISBI 2012/Val-Labels/train-labels-*.tif",
        length=6, is_pad=args.pad, eval=True, totensor=True)

# num of workers can represent the number of cores in cpu, pinned memory is page-locked memory
# disable it  if system freezes, or swap is used a lot
# https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
# batchsize is 1 for validation, to get a single output for loss and not a mean

trainloader = dl.DataLoader(trainset, batch_size=args.minibatchsize,  num_workers=args.workers, pin_memory=True)
valloader = dl.DataLoader(valset, batch_size=1,  num_workers=args.workers, pin_memory=True)

# 3: Training cycle forward, backward , update

# load the model if set
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.last_epoch = args.start_epoch
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# print some info for console
print'Dataset      : ' + str(args.data)
print'Start Epoch  : ' + str(args.start_epoch)
print'End Epoch    : ' + str(args.epochs)
print'Learning rate: ' + str(args.lr)
print'Momentum     : ' + str(args.momentum)
print'Weight decay : ' + str(args.weight_decay)
print'Use padding  : ' + str(args.pad)

#  save a txt file with the console info
if args.txt:
    with open("Info_lr_" + str(args.lr) + "_wd_" + str(args.weight_decay) + ".txt", "a") as myfile:
        myfile.write('Dataset      : ' + str(args.data))
        myfile.write('\n')
        myfile.write('Start Epoch  : ' + str(args.start_epoch))
        myfile.write('\n')
        myfile.write('End Epoch    : ' + str(args.epochs))
        myfile.write('\n')
        myfile.write('Learning rate: ' + str(args.lr))
        myfile.write('\n')
        myfile.write('Momentum     : ' + str(args.momentum))
        myfile.write('\n')
        myfile.write('Weight decay : ' + str(args.weight_decay))
        myfile.write('\n')
        myfile.write('Use padding  : ' + str(args.pad))
        myfile.write('\n')
        myfile.close()

if args.evaluate:
    print " avg loss: " + str(eval(valloader, model, criterion, True))
else:
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        train_loss = train(trainloader, model, criterion, optimizer, epoch)
        val_loss = eval(valloader, model, criterion, False)
        end_time = time.time()

        print('Epoch [%5d] train_loss: %.4f val_loss: %.4f loop time: %.5f' %
              (epoch + 1, train_loss, val_loss, end_time - start_time))
        if args.txt:
            with open("Info_lr_" + str(args.lr) + "_wd_" + str(args.weight_decay) + ".txt", "a") as myfile:
                myfile.write('Epoche [%5d] train_loss: %.4f val_loss: %.4f loop time: %.5f' %
                             (epoch + 1, train_loss, val_loss, end_time - start_time))
                myfile.write('\n')
                myfile.close()

        # see info at criterion above
        # scheduler.step(val_loss)
        # Data Augmentation
        # 50% change to flip or random rotate, same for whole batch
        # change every epoch
        # starting epoche with no flipping and rotation
        trainloader.dataset.rand_vflip = random.random() < 0.5
        trainloader.dataset.rand_hflip = random.random() < 0.5
        #rotate image
        #trainloader.dataset.rand_rotate = random.random() < 0.5
        #trainloader.dataset.angle = random.uniform(-180, 180)

        #save best loss
        is_best_loss = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        # save model
        if (epoch + 1) % args.epochsave == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best_loss, filename='checkpoint.'+ str(args.lr) + "wd" + str(args.weight_decay) + '.pth.tar')
print "*****   End  Programm   *****"



