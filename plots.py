import os
import torch
import matplotlib.pyplot as plt
import numpy as np
resume = 'checkpoint.0.001wd0.pth.tar'
if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    t = np.arange(1, start_epoch + 1)
    x = np.asarray(train_loss)
    y = np.asarray(val_loss)
    plt.plot(t, x, label="train loss")
    plt.title('train loss CE Loss')
    plt.legend()
    plt.savefig('CEtrain.jpg')
    plt.show()
    plt.plot(t, y, label="val loss")
    plt.legend()
    plt.title('val loss CE Loss')
    plt.savefig('CEtval.jpg')
    plt.show()

    plt.plot(t, x, label="train loss")
    plt.plot(t, y, label="val loss")
    plt.title('loss CE')
    plt.legend()
    plt.savefig('CEloss.jpg')
    plt.show()




