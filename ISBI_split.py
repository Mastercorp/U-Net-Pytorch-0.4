from PIL import Image
import os
from skimage.transform import AffineTransform, warp
import numpy as np

# For the Dataset register at : http://brainiac2.mit.edu/isbi_challenge/
# Download the corresponding data files

def affinetrans(img, x, y):
    tform = AffineTransform(translation=(x, y))
    imgaug = np.asarray(img)
    imgaug = Image.fromarray((warp(imgaug, tform, mode='reflect') * 255).astype(np.uint8))
    return imgaug


directory = './ISBI 2012/Train-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Val-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Train-Labels/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Val-Labels/'
if not os.path.exists(directory):
    os.makedirs(directory)


imgvolume = Image.open('./train-volume.tif')
imglabel = Image.open('./train-labels.tif')

imgindex = 0
for i in range(30):
    try:
        imgvolume.seek(i)
        imglabel.seek(i)

        if i % 5 == 0:
            imgvolume.save('./ISBI 2012/Val-Volume/train-volume-%s.tif' % (imgindex,))
            imglabel.save('./ISBI 2012/Val-Labels/train-labels-%s.tif' % (imgindex,))

        else:
            imgvolume.save('./ISBI 2012/Train-Volume/train-volume-%s.tif' % (imgindex,))
            imglabel.save('./ISBI 2012/Train-Labels/train-labels-%s.tif' % (imgindex,))

        imgindex = imgindex + 1
        x = 3
        y = 3
        for x1 in range(-x, x + 1, 2):
            for y1 in range(-y, y + 1, 2):
                if i % 5 == 0:
                    affinetrans(imgvolume, x1, y1).save('./ISBI 2012/Val-Volume/train-volume-%s.tif' % (imgindex,))
                    affinetrans(imglabel, x1, y1).save('./ISBI 2012/Val-Labels/train-labels-%s.tif' % (imgindex,))
                else:
                    affinetrans(imgvolume, x1, y1).save('./ISBI 2012/Train-Volume/train-volume-%s.tif' % (imgindex,))
                    affinetrans(imglabel, x1, y1).save('./ISBI 2012/Train-Labels/train-labels-%s.tif' % (imgindex,))
                imgindex = imgindex + 1
    except EOFError:
        break


img = Image.open('./test-volume.tif')
directory = './ISBI 2012/Test-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)
for i in range(30):
    try:
        img.seek(i)
        img.save('./ISBI 2012/Test-Volume/test-volume-%s.tif' % (i,))
    except EOFError:
        break
