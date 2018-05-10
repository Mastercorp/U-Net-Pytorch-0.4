# U-Net-Pytorch-0.4
U-Net Implementation for Pytorch 0.4

Custom Dataloader for the ISBI 2012 Challenge Dataset + Data Augmentation   
The dataset is not included in this repro   

## How to use:

First download the ISBI 2012 Dataset in your folder. Start ISBI_split.py which generates a folder structure.
for all options run:
main.py - h

for a simple start use:
main.py ISBI2012 -s -p -txt

-s    saves the first image each epoche
-p    uses padding to stop the reduction of image size caused by 3x3 convolutions
-txt  save information about the used settings and losses each epoch


the txt file looks for example like this:
Dataset      : ISBI2012
Start Epoch  : 0
End Epoch    : 100
Learning rate: 0.001
Momentum     : 0.99
Weight decay : 0
Use padding  : True
Epoche [    1] train_loss: 0.4911 val_loss: 0.4643 loop time: 9.96429
Epoche [    2] train_loss: 0.4630 val_loss: 0.5017 loop time: 5.41091
Epoche [    3] train_loss: 0.4460 val_loss: 0.4637 loop time: 5.45516



## Requirements
Tested on:      
Ubuntu 16.04      
Geforce GTX 1070 8GB Nvidia Driver 390.48 CUDA 9.1   
Python 2.7.14   
Pytorch 0.4   
conda install pytorch torchvision cuda91 -c pytorch   

U-Net: Convolutional Networks for Biomedical Image Segmentation   
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/   
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.   

ISBI 2012 Segmentation Challenge   
http://brainiac2.mit.edu/isbi_challenge/home   
Ignacio Arganda-Carreras, Srinivas C. Turaga, Daniel R. Berger, Dan Ciresan, Alessandro Giusti, Luca M. Gambardella, Jürgen Schmidhuber, Dmtry Laptev, Sarversh Dwivedi, Joachim M. Buhmann, Ting Liu, Mojtaba Seyedhosseini, Tolga Tasdizen, Lee Kamentsky, Radim Burget, Vaclav Uher, Xiao Tan, Chanming Sun, Tuan D. Pham, Eran Bas, Mustafa G. Uzunbas, Albert Cardona, Johannes Schindelin, and H. Sebastian Seung. Crowdsourcing the creation of image segmentation algorithms for connectomics. Frontiers in Neuroanatomy, vol. 9, no. 142, 2015.   
