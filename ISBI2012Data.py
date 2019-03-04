import glob
from torch.utils import data
from PIL import Image
import torchvision
import random
import dataaug

import paramssacred

ex = paramssacred.ex


class ISBIDataset(data.Dataset):

    def __init__(self, gloob_dir_train, gloob_dir_label, length, is_pad, evaluate, totensor):
        self.gloob_dir_train = gloob_dir_train
        self.gloob_dir_label = gloob_dir_label
        self.length = length
        self.crop = torchvision.transforms.CenterCrop(512)
        self.crop_nopad = torchvision.transforms.CenterCrop(324)
        self.is_pad = is_pad
        self.evaluate = evaluate
        self.totensor = totensor
        self.changetotensor = torchvision.transforms.ToTensor()

        self.trainfiles = sorted(glob.glob(self.gloob_dir_train),
                                 key=lambda name: int(name[self.gloob_dir_train.rfind('*'):
                                                           -(len(self.gloob_dir_train) - self.gloob_dir_train.rfind(
                                                               '.'))]))

        self.labelfiles = sorted(glob.glob(self.gloob_dir_label),
                                 key=lambda name: int(name[self.gloob_dir_label.rfind('*'):
                                                           -(len(self.gloob_dir_label) - self.gloob_dir_label.rfind(
                                                               '.'))]))

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    @ex.capture
    def __getitem__(self, index, augment):
        'Generates one sample of data'
        # files are sorted depending the last number in their filename
        # for example : "./ISBI 2012/Train-Volume/train-volume-*.tif"

        trainimg1 = Image.open(self.trainfiles[index]).convert("L")
        trainlabel1 = Image.open(self.labelfiles[index]).convert("L")
        trainimg = trainimg1
        trainlabel = trainlabel1
        if not self.evaluate:

            # rotate input image
            angle = random.randint(0, augment["angleparts"] - 1) * (360.0 / augment["angleparts"])
            trainimg1 = dataaug.rotate_img(trainimg1, angle)

            trainlabel1 = dataaug.rotate_img(trainlabel1, angle)
            trainlabel1 = dataaug.maplabel(trainlabel1, 127, 0, 255)

            if random.random() < 0.5:
                trainlabel1 = dataaug.flip_lr(trainlabel1)
                trainimg1 = dataaug.flip_lr(trainimg1)

            if random.random() < 0.5:
                trainlabel1 = dataaug.flip_tb(trainlabel1)
                trainimg1 = dataaug.flip_tb(trainimg1)

            if random.random() < 2:

                sigma = random.randint(augment["sigmamin"], augment["sigmamax"])
                # to shift in gradient direction -, else it shifts in the other direction
                scalex = random.randint(augment["minscale"], augment["maxscale"])
                scaley = random.randint(augment["minscale"], augment["maxscale"])
                if random.random() < 0.5:
                    scalex = scalex * -1
                if random.random() < 0.5:
                    scaley = scaley * -1

                trainimg, indices = dataaug.gradtrans(trainimg1, scalex, scaley, sigma)

                trainlabel1 = dataaug.indextrans(trainlabel1, indices)
                trainlabel = dataaug.maplabel(trainlabel1, 127, 0, 255)



        # when padding is used, dont crop the label image
        if not self.is_pad:
            trainlabel = self.crop_nopad(trainlabel)

        if self.totensor:
            # test if NLL needs long
            trainlabel = self.changetotensor(trainlabel).long()
            trainimg = self.changetotensor(trainimg)

        return trainimg, trainlabel
