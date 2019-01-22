from PIL import Image
import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-img', metavar='IMAGEFILE',
                    help='choose the image file which should be augmented')

parser.add_argument('-smin', metavar='SIGMAMIN', default=70,
                    help='minimal sigma value for gaussian blur to reduce noise (default:70)')
parser.add_argument('-stepsize', metavar='INTEGER', default=10,
                    help='stepsize between sigma values (default:10)')
parser.add_argument('-smax', metavar='SIGMAMAX', default=100,
                    help='maximal sigma value for gaussian blur to reduce noise (default:100)')
parser.add_argument('-iter', metavar='INTEGER', default = 10,
                    help='number of iterations in each sigma setting')
parser.add_argument('-scale', metavar='INTEGER', default=1000,
                    help='Scaling factor which is applied to the gradients  (default:1000)')


args = parser.parse_args()
image = Image.open(args.img).convert("L")

## GRADIENT Augmentation

imagestart = 1
for t in range(args.smin, args.smax + args.stepsize, args.stepsize):

    sigma = t
    print sigma
    # flip image upside down ( image 0 0 is in the bottom left, while matrix start is in top left.
    flipimg = np.flipud(np.asarray(image))

    dy = np.gradient(flipimg.astype('float32'), axis=0)
    dx = np.gradient(flipimg.astype('float32'), axis=1)

    # normalize dx and dy
    dx /= np.max(np.abs(dx))
    dy /= np.max(np.abs(dy))
    for it in range(args.iter):

        # to shift in the opposite direction of the gradient, scale needs to be negative
        # else it shifts in the gradient direction
        # values are scaled with 1000. Depending on the task, this value needs to adjust
        scalex = random.randint(1, 10) * args.scale
        scaley = random.randint(1, 10) * args.scale

        #make the shifting process in both directions random
        if random.random() <0.5:
            scalex = scalex *-1
        if random.random() < 0.5:
            scaley = scaley * -1

        dxconv = gaussian_filter(dx, sigma, order=0, mode='mirror', truncate=3) * -scalex
        dyconv = gaussian_filter(dy, sigma, order=0, mode='mirror', truncate=3) * -scaley


        x, y = np.meshgrid(np.arange(flipimg.shape[1]), np.arange(flipimg.shape[0]), indexing='xy')

        indices = [np.reshape(y + dyconv, (-1, 1)), np.reshape(x + dxconv, (-1, 1))]
        trainimg = Image.fromarray(
            map_coordinates(flipimg, indices, order=1, mode='mirror').reshape((flipimg.shape[0], flipimg.shape[1])))

        #flip image again
        trainimg = trainimg.transpose(Image.FLIP_TOP_BOTTOM)
        trainimg.save('outfile-%s.tif' % (imagestart,))
        imagestart += 1
