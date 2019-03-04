import torch.utils.data.dataloader as dl
import ISBI2012Data as ISBI
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import numpy as np
from PIL import Image
import sys

import time
import model as md

import paramssacred

ex = paramssacred.ex


# saves the model with learning rate and weight decay
@ex.capture
def save_checkpoint(state, filename, params):
    torch.save(state, str(params["savedir"]) + filename)


@ex.capture
def train(trainloader, model, criterion, optimizer, epoch, params):
    model.train()
    loss_sum = 0
    for i, data in enumerate(trainloader):
        # when using enumerate on trainloader, the whole set is used ( all 22 images, but in shuffled order
        # i want to randomly use my data.
        # for i, data in enumerate(trainloader):

        # Forces the batch to use 30 samples and reduce overfitting when shuffle = True
        # get train and label data
        # train, label = data

        # the first return value, which is an index.
        train, label = data
        # put on gpu or cpu
        train = train.to(udevice)
        # label is of type TensorLong
        label = label.to(udevice)
        # for the CrossEntropyLoss:
        # outputs needs to be of size: Minibatch, classsize, dim 1, dim 2 , ...
        # outputs  are 2 classes with 2d images. channelsize = class size
        # label needs to be of format: Minibatch, dim 1, dim 2, ...
        # I cut the channel info for it to work, because it is only a 2d image.
        # As an alternative, one could add 1 channel for class in train, than label does not need any change
        # label normally looks like: ( minibatchsize, class, width, height )

        label = label.view(label.size(0), label.size(2), label.size(3))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # if padding is true, one row and one column is added at left, right, top and bottom at each convolution
        # to maintain the original size of the image
        # (1, 2, 512, 512)
        outputs = model(train, padding=params["padding"])


        if params["saveimages"] and (i == 0):
            save_images(outputs, str(params["savedir"]) + str(params["learningrate"]) + '/', epoch, index=i)


        loss = criterion(torch.log(outputs), label)

        loss.backward()

        optimizer.step()

        running_loss = loss.item()
        loss_sum = loss_sum + running_loss

        # delete all references to variables:
        # https://discuss.pytorch.org/t/tensor-to-vari
        del outputs, train, label, loss
    loss_avg = loss_sum / (i + 1)

    return loss_avg


@ex.capture
def evaluate(valloader, model, criterion, save_image, params):
    # switch the model to eval mode ( important for dropout layers or batchnorm layers )
    model.eval()
    loss_sum = 0
    for i, data in enumerate(valloader):
        # get train and label data
        val, label = data
        # put on gpu or cpu
        val = val.to(udevice)
        # label is of type TensorLong

        label = label.to(udevice)

        # for the CrossEntropyLoss:
        # outputs needs to be of size: Minibatch, classsize, dim 1, dim 2 , ...
        # outputs  are 2 classes with 2d images. Channelsize = class size
        # label needs to be of format: Minibatch, dim 1, dim 2, ...
        # i cut the channel info for it to work, because it is only a 2d image.
        # as an alternative, one could add 1 channel for class in train, than label does not need any change
        # label normally looks like: ( minibatchsize, 1, width, height )
        label = label.view(label.size(0), label.size(2), label.size(3))

        # forward + backward + optimize
        outputs = model(val, padding=params["padding"])

        loss = criterion(torch.log(outputs), label)

        running_loss = loss.item()
        loss_sum = loss_sum + running_loss

        if save_image:
            save_images_eval(outputs, './eval/', index=i + 1)

        del outputs, val, label, loss

    loss_avg = loss_sum / (i + 1)
    return loss_avg


def save_images_eval(outputs, directory, index):
    # copy first image in outputs back to cpu and save it

    y = outputs[0][1][:][:].cpu().detach().numpy()

    # convert image to save it properly
    # for visibility in grayscale
    # for competition comment this 2 lines out
    # probabilities in the range of 0-1

    if not os.path.exists(directory):
        os.makedirs(directory)
    y = Image.fromarray(y)
    name = directory + "%03d.tif" % index
    y.save(name)


def save_images(outputs, directory, epoch, index):
    # copy first image in outputs back to cpu and save it
    x = outputs[0][0][:][:].cpu().detach().numpy()
    y = outputs[0][1][:][:].cpu().detach().numpy()

    x = (x * 255).astype(np.uint8)
    y = (y * 255).astype(np.uint8)

    x = Image.fromarray(x)
    if not os.path.exists(directory):
        os.makedirs(directory)
    x.save(directory + 'class1_' + str(epoch) + '_image_' + str(index) + '.tif')
    y = Image.fromarray(y)
    y.save(directory + 'class2_' + str(epoch) + '_image_' + str(index) + '.tif')


torch.backends.cudnn.deterministic = True
# check if cuda is available
udevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ex.automain
def my_main(params):
    print ("***** Starting Programm *****")
    # initialize some variables
    train_loss = []
    val_loss = []
    best_loss = 10
    global udevice

    if params["cpu"]:
        udevice = torch.device("cpu")

    # use cudnn for better speed, if available
    if udevice.type == "cuda":
        cudnn.benchmark = True

    # 1: design model
    model = md.Unet().to(udevice)

    # 2: Construct loss and optimizer

    # Using a softmax layer at the end, applying the log and using NLLoss()
    # has the same loss as using no softmax layer, and calculating the CrossEntropyLoss()
    # the difference is in the output image of the model.
    # If you want to use the CrossEntropyLoss(), remove the softmax layer, and  the torch.log() at the loss

    criterion = nn.NLLLoss(weight=torch.tensor(params["classweight"])).to(udevice)

    optimizer = optim.SGD(model.parameters(), lr=params["learningrate"],
                          momentum=params["momentum"], weight_decay=params["weightdecay"])

    # Reduce learning rate when a metric has stopped improving, needs to be activated in epoch too
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, verbose=True)

    # load the ISBI 2012 training data
    # the CTC2015 Datasetloader is not finished yet
    # The length of the Dataset has to be set by yourself
    # gloob_dir_train, gloob_dir_label, length, is_pad, eval, totensor):

    trainset = ISBI.ISBIDataset(
        "./ISBI 2012/Train-Volume/train-volume-*.tif", "./ISBI 2012/Train-Labels/train-labels-*.tif",
        length=22, is_pad=params["padding"], evaluate=False, totensor=True)

    if not params["evaluate"]:
        valset = ISBI.ISBIDataset(
            "./ISBI 2012/Val-Volume/train-volume-*.tif", "./ISBI 2012/Val-Labels/train-labels-*.tif",
            length=8, is_pad=params["padding"], evaluate=True, totensor=True)
    else:
        valset = ISBI.ISBIDataset(
            "./ISBI 2012/Test-Volume/test-volume-*.tif", "./ISBI 2012/Test-Volume/test-volume-*.tif",
            length=30, is_pad=params["padding"], evaluate=True, totensor=True)

    # num of workers can represent the number of cores in cpu, pinned memory is page-locked memory
    # disable it  if system freezes, or swap is used a lot
    # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
    # batchsize is 1 for validation, to get a single output for loss and not a mean

    # shuffle input data with replacement
    # in Pytorch 0.41 only WeightedRandomSampler can do this
    # Therefore, create list with ones, which corresponds to the length of trainset.
    # take 30 samples per epoch.
    listones = [1] * trainset.length
    randomsampler = torch.utils.data.WeightedRandomSampler(listones, 30, True)
    trainloader = dl.DataLoader(trainset, sampler=randomsampler, batch_size=params["batch_size"],
                                num_workers=params["workers"], pin_memory=True)
    valloader = dl.DataLoader(valset, batch_size=1, num_workers=params["workers"], pin_memory=True)

    # 3: Training cycle forward, backward , update

    # load the model if set
    if params["resume"]:
        if os.path.isfile(params["resume"]):
            print("=> loading checkpoint '{}'".format(params["resume"]))
            checkpoint = torch.load(params["resume"])
            params["startepoch"] = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
            # scheduler.last_epoch = args.start_epoch
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(params["resume"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(params["resume"]))
            sys.exit(0)

    # print some info for console
    print('Dataset      :  ISBI 2012')
    print('Start Epoch  : ' + str(params["startepoch"]))
    print('End Epoch    : ' + str(params["epochs"]))
    print('Learning rate: ' + str(params["learningrate"]))
    print('Momentum     : ' + str(params["momentum"]))
    print('Weight decay : ' + str(params["weightdecay"]))
    print('Use padding  : ' + str(params["padding"]))

    #  save a txt file with the console info
    if params["txtinfo"] and not params["evaluate"]:
        with open(str(params["savedir"]) + "txtinfo.txt", "a") as myfile:
            myfile.write('Dataset      : ISBI2012')
            myfile.write('\n')
            myfile.write('Start Epoch  : ' + str(params["startepoch"]))
            myfile.write('\n')
            myfile.write('End Epoch    : ' + str(params["epochs"]))
            myfile.write('\n')
            myfile.write('Learning rate: ' + str(params["learningrate"]))
            myfile.write('\n')
            myfile.write('Momentum     : ' + str(params["momentum"]))
            myfile.write('\n')
            myfile.write('Weight decay : ' + str(params["weightdecay"]))
            myfile.write('\n')
            myfile.write('Use padding  : ' + str(params["padding"]))
            myfile.write('\n')
            myfile.close()

    if params["evaluate"]:
        print(" avg loss: " + str(evaluate(valloader, model, criterion, True)))
    else:
        print ("***** Start Training *****")

        breakloss = 0
        # val loss and train loss are initialized with 0
        for epoch in range(params["startepoch"], params["epochs"]):
            start_time = time.time()

            train_loss.append(train(trainloader, model, criterion, optimizer, epoch))
            val_loss.append(evaluate(valloader, model, criterion, False))
            end_time = time.time()

            print('Epoch [%5d] train_loss: %.4f val_loss: %.4f loop time: %.5f' %
                  (epoch + 1, train_loss[epoch], val_loss[epoch], end_time - start_time))
            if params["txtinfo"]:
                with open(str(params["savedir"]) + "txtinfo.txt", "a") as myfile:
                    myfile.write('Epoche [%5d] train_loss: %.4f val_loss: %.4f loop time: %.5f' %
                                 (epoch + 1, train_loss[epoch], val_loss[epoch], end_time - start_time))
                    myfile.write('\n')
                    myfile.close()

            # see info at criterion above
            # scheduler.step(val_loss)
            # Data Augmentation

            if 0.6931 < train_loss[epoch] < 0.6932:
                breakloss += 1
                if breakloss > 7:
                    sys.exit()
            else:
                breakloss = 0

            # save best loss
            is_best_loss = val_loss[epoch] < best_loss
            best_loss = min(val_loss[epoch], best_loss)

            if is_best_loss:
                ex.log_scalar('best_epoch', epoch + 1)

            ex.log_scalar('val_loss', val_loss[epoch])
            ex.log_scalar('train_loss', train_loss[epoch])

            # save model
            filename = ""
            if is_best_loss:
                filename = 'best_loss.pth.tar'
            else:
                filename = 'current.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            }, filename=filename)

        print ("*****   End  Programm   *****")
