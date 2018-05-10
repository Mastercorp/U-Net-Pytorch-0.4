import glob
from torch.utils import data
from PIL import Image
import torchvision
import numpy as np

class ISBIDataset(data.Dataset):

    def __init__(self, gloob_dir_train, gloob_dir_label, length, is_pad, eval, totensor):
        self.gloob_dir_train = gloob_dir_train
        self.gloob_dir_label = gloob_dir_label
        self.length = length
        self.crop = torchvision.transforms.CenterCrop(512)
        self.crop_nopad = torchvision.transforms.CenterCrop(324)
        self.is_pad = is_pad
        self.eval = eval
        self.totensor = totensor
        self.changetotensor = torchvision.transforms.ToTensor()

        self.rand_vflip = False
        self.rand_hflip = False
        self.rand_rotate = False
        self.angle = 0

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        'Generates one sample of data'
        # files are sorted depending the last number in their filename
        # for example : "./ISBI 2012/Train-Volume/train-volume-*.tif"
        trainfiles = sorted(glob.glob(self.gloob_dir_train),
                            key=lambda name: int(name[self.gloob_dir_train.rfind('*'):
                                                      -(len(self.gloob_dir_train) - self.gloob_dir_train.rfind('.'))]))

        labelfiles = sorted(glob.glob(self.gloob_dir_label),
                            key=lambda name: int(name[self.gloob_dir_label.rfind('*'):
                                                      -(len(self.gloob_dir_label) - self.gloob_dir_label.rfind('.'))]))

        trainimg = Image.open(trainfiles[index])
        trainlabel = Image.open(labelfiles[index])


        if not self.eval:
            if self.rand_vflip:
                trainlabel = trainlabel.transpose(Image.FLIP_LEFT_RIGHT)
                trainimg = trainimg.transpose(Image.FLIP_LEFT_RIGHT)

            if self.rand_hflip:
                trainlabel = trainlabel.transpose(Image.FLIP_TOP_BOTTOM)
                trainimg = trainimg.transpose(Image.FLIP_TOP_BOTTOM)

            if self.rand_rotate:
                # Add padding to the image to remove black boarders when rotating
                # image is croped to true size later.
                trainimg = Image.fromarray(np.pad(np.asarray(trainimg), ((107, 107), (107, 107)), 'reflect'))
                trainlabel = Image.fromarray(np.pad(np.asarray(trainlabel), ((107, 107), (107, 107)), 'reflect'))

                trainlabel = trainlabel.rotate(self.angle)
                trainimg = trainimg.rotate(self.angle)
                # crop rotated image to true size
                trainlabel = self.crop(trainlabel)
                trainimg = self.crop(trainimg)


        # when padding is used, dont crop the label image
        if not self.is_pad:
            trainlabel = self.crop_nopad(trainlabel)

        if self.totensor:
            trainlabel = self.changetotensor(trainlabel).long()
            trainimg = self.changetotensor(trainimg)

        return trainimg, trainlabel
