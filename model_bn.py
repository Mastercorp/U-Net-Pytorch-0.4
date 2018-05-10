import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1 MODEL
class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        # All layers which have weights are created and initlialitzed in init.
        # parameterless modules are used in functional style F. in forward
        # (object version of parameterless modules can be created with nn. init too )

        # https://pytorch.org/docs/master/nn.html#conv2d
        # in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)

        # https://pytorch.org/docs/master/nn.html#batchnorm2d
        # num_features/channels, eps, momentum, affine, track_running_stats
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)

        # https://pytorch.org/docs/master/nn.html#maxpool2d
        # kernel_size, stride, padding, dilation, return_indices, ceil_mode
        self.maxPool1 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxPool2 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=0)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=0)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.maxPool3 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=0)
        self.conv7_bn = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.maxPool4 = nn.MaxPool2d(2, stride=2, padding=0)

        self.conv9 = nn.Conv2d(512, 1024, 3, stride=1, padding=0)
        self.conv9_bn = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=0)
        self.conv10_bn = nn.BatchNorm2d(1024)

        # https://pytorch.org/docs/master/nn.html#convtranspose2d
        # in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation
        self.upsampconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0)

        self.conv11 = nn.Conv2d(1024, 512, 3, stride=1, padding=0)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv12_bn = nn.BatchNorm2d(512)

        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)

        self.conv13 = nn.Conv2d(512, 256, 3, stride=1, padding=0)
        self.conv13_bn = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 256, 3, stride=1, padding=0)
        self.conv14_bn = nn.BatchNorm2d(256)

        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)

        self.conv15 = nn.Conv2d(256, 128, 3, stride=1, padding=0)
        self.conv15_bn = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv16_bn = nn.BatchNorm2d(128)

        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)

        self.conv17 = nn.Conv2d(128, 64, 3, stride=1, padding=0)
        self.conv17_bn = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv18_bn = nn.BatchNorm2d(64)

        self.conv19 = nn.Conv2d(64, 2, 1, stride=1, padding=0)
        self.conv19_bn = nn.BatchNorm2d(2)
        self.softmax = nn.Softmax2d()

        # weights can be initialized here:
        # for example:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # force float division, therefore use 2.0
                # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
                # https://arxiv.org/abs/1502.01852
                # a rectifying linear unit is zero for half of its input,
                # so you need to double the size of weight variance to keep the signals variance constant.
                # xavier would be: scalefactor * sqrt(2/ (inchannels + outchannels )
                std = math.sqrt(2.0/(m.kernel_size[0]*m.kernel_size[0]*m.in_channels))
                nn.init.normal_(m.weight, std=std)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     print
            #    # nn.init.constant_(m.weight, 1)
            #     #nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.ConvTranspose2d):
            #     print
            #    # nn.init.xavier_normal_(m.weight, 1)
            #    # do max pooling layers have weight? maybe can add bias.
            # elif isinstance(m, nn.MaxPool2d):
            #     print
            #     #nn.init.xavier_normal_(m.weight)

    def forward(self, x, padding=False):

        # https://pytorch.org/docs/master/nn.html#torch.nn.ReLU
        # https://pytorch.org/docs/master/nn.html#id26 F.relu
        # input, inplace
        # https://pytorch.org/docs/master/nn.html#torch.nn.functional.pad
        # input, pad , mode
        padmode = 'reflect'
        if padding:
            pad = (1, 1, 1, 1)
        else:
            pad = (0, 0, 0, 0)

        x = F.relu(self.conv1_bn(self.conv1(F.pad(x, pad, padmode))))
        x = F.relu(self.conv2_bn(self.conv2(F.pad(x, pad, padmode))))
        # save result for combination later
        x_copy1_2 = x
        x = self.maxPool1(x)

        x = F.relu(self.conv3_bn(self.conv3(F.pad(x, pad, padmode))))
        x = F.relu(self.conv4_bn(self.conv4(F.pad(x, pad, padmode))))
        x_copy3_4 = x
        x = self.maxPool2(x)

        x = F.relu(self.conv5_bn(self.conv5(F.pad(x, pad, padmode))))
        x = F.relu(self.conv6_bn(self.conv6(F.pad(x, pad, padmode))))
        x_copy5_6 = x
        x = self.maxPool3(x)

        x = F.relu(self.conv7_bn(self.conv7(F.pad(x, pad, padmode))))
        x = F.relu(self.conv8_bn(self.conv8(F.pad(x, pad, padmode))))
        # input, probability of an element to be zero-ed
        # https://pytorch.org/docs/master/nn.html#dropout
        x = F.dropout(x, 0.5)
        x_copy7_8 = x
        x = self.maxPool4(x)

        x = F.relu(self.conv9_bn(self.conv9(F.pad(x, pad, padmode))))
        x = F.relu(self.conv10_bn(self.conv10(F.pad(x, pad, padmode))))
        x = F.dropout(x, 0.5)
        x = F.relu(self.upsampconv1(x))

        x = self.crop_and_concat(x, x_copy7_8)

        x = F.relu(self.conv11_bn(self.conv11(F.pad(x, pad, padmode))))
        x = F.relu(self.conv12_bn(self.conv12(F.pad(x, pad, padmode))))

        x = F.relu(self.upsampconv2(x))

        x = self.crop_and_concat(x, x_copy5_6)

        x = F.relu(self.conv13_bn(self.conv13(F.pad(x, pad, padmode))))
        x = F.relu(self.conv14_bn(self.conv14(F.pad(x, pad, padmode))))

        x = F.relu(self.upsampconv3(x))

        x = self.crop_and_concat(x, x_copy3_4)

        x = F.relu(self.conv15_bn(self.conv15(F.pad(x, pad, padmode))))
        x = F.relu(self.conv16_bn(self.conv16(F.pad(x, pad, padmode))))

        x = F.relu(self.upsampconv4(x))

        x = self.crop_and_concat(x, x_copy1_2)

        x = F.relu(self.conv17_bn(self.conv17(F.pad(x, pad, padmode))))
        x = F.relu(self.conv18_bn(self.conv18(F.pad(x, pad, padmode))))

        x = F.relu(self.conv19_bn(self.conv19(x)))

        x = self.softmax(x)
        return x

    # when no padding is used, the upsampled image gets smaller
    # to copy a bigger image to the corresponding layer, it needs to be cropped
    def crop_and_concat(self, upsampled, bypass):
        # Python 2 / Integer division ( if int intputs ), // integer division
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        d = c
        # checks if bypass.size() is odd
        # if input image is 512, at   x = self.crop_and_concat(x, x_copy5_6)
        # x_copy5_6 is 121*121
        # therefore cut one more row and column
        if (bypass.size()[2] & 1) == 1:
            d = c + 1
            # padleft padright padtop padbottom
        bypass = F.pad(bypass, (-c, -d, -c, -d))
        return torch.cat((bypass, upsampled), 1)
