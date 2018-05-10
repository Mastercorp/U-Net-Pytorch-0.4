from PIL import Image
import os

# For the Dataset register at : http://brainiac2.mit.edu/isbi_challenge/
# Download the corresponding data files
# Only 30 images are available with ground truth
# 6 Images are used for validation and are put into a seperate folder
img = Image.open('./train-volume.tif')

directory = './ISBI 2012/Train-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Val-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)
for i in range(30):
    try:
        img.seek(i)
        if i % 5 == 0:
            img.save('./ISBI 2012/Val-Volume/train-volume-%s.tif' % (i,))
        else:
            img.save('./ISBI 2012/Train-Volume/train-volume-%s.tif' % (i,))
    except EOFError:
        break
img = Image.open('./train-labels.tif')
directory = './ISBI 2012/Train-Labels/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Val-Labels/'
if not os.path.exists(directory):
    os.makedirs(directory)
for i in range(30):
    try:
        img.seek(i)
        if i % 5 == 0:
            img.save('./ISBI 2012/Val-Labels/train-labels-%s.tif' % (i,))
        else:
            img.save('./ISBI 2012/Train-Labels/train-labels-%s.tif' % (i,))
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