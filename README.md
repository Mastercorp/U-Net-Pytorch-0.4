# Introduction
U-Net-Pytorch-0.4 is a custom U-Net implementation in python 2.7 for Pytorch 0.4.
Furthermore, a custom dataloader is introduced, which can load the ISBI 2012 Dataset.

Details about the U-Net network can be found on the U-Net [project page](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
The implementation in this repository is tested on Ubuntu 16.04 with Python 2.7    

# License
The implementation is freely available under the MIT License,
meaning that you can basically do with the code whatever you want.



# Dependencies
* Pytorch 0.4, Python 2.7, ( CUDA 9.1 for GPU acceleration )   
 



# Building
No building is required, just clone or download the github project in a directory. The programm is tested on Ubuntu 16.04 with a Geforce GTX 1070 8GB Nvidia Driver 390.48 CUDA 9.1, Python 2.7.14 and Pytorch 0.4  


# Usage
/python should start python 2.7 . You can check the version with /python --version
```
usage: /python main.py [-h] [-mbs N] [-j N] [--epochs N] [-lr LR] [--momentum M]
               [-es M] [--weight-decay W] [-r PATH] [-e] [-s] [-c] [-p] [-txt]
               [-bn]
               dataset

```
## positional arguments
* `dataset` ISBI2012

## Optional arguments
*  `-h, --help`            show the help message
*  `-mbs N, --mini-batch-size N`
                        mini batch size (default: 1). For 8k memory on gpu,
                        minibatchsize of 2-3 possible
*  `-j N, --workers N`     number of data loading workers (default: 2)
*  `--epochs N`            number of total epochs to run (default: 20)
*  `-lr LR`                initial learning rate (default: 0.001)
*  `--momentum M`          momentum (default: 0.99)
*  `-es M, --epochsave M`  save model every M epoch (default: 1)
*  `--weight-decay W, -wd W`
                        weight decay (L2 penalty ) (default:0)
*  `-r PATH, --resume PATH`
                        relative path to latest checkpoint, load all needed data to resume the network (default: none)   
*  `-e, --evaluate`        evaluate model on validation set
*  `-s, --save-images`     save the first image of output each epoche
*  `-c, --cpu`             use cpu instead of gpu
*  `-p, --pad`             use padding at each 3x3 convolution to maintain image
                        size
*  `-txt`                  save console output in txt
*  `-bn`                   use u-net with batchnorm layers added after each
                        convolution




## Examples


Use the ISBI2012 dataset and run for 600 epochs. Use padding at each 3x3 convolution and save information about the used settings and losses each epoch in a txt file.
```
python main.py ISBI2012 --epochs 600 -p -txt
```
the txt file looks like this:
```
Dataset : ISBI2012
Start Epoch : 0
End Epoch : 100
Learning rate: 0.001
Momentum : 0.99
Weight decay : 0
Use padding : True
Epoche [ 1] train_loss: 0.4911 val_loss: 0.4643 loop time: 9.96429
Epoche [ 2] train_loss: 0.4630 val_loss: 0.5017 loop time: 5.41091
Epoche [ 3] train_loss: 0.4460 val_loss: 0.4637 loop time: 5.45516
```

## Donate
Bitcoin: 1NE7tpCaHXMG3VP2oQrx1L53MEPnAp39xM  
Litecoin: LLCmNPWBt8TxnNuaTWfyDdajXM5rZPuzsT  
Verge   : DN5QsxVaFLykVykGNFqRdGFvi5zg2zz3Rq   
Reddcoin: Rsjofwt2TNu6Gf6eQdzyUhJe2J6vLsKoBq  
Dogecoin: DM13fiivexaZf35HTQ7AKBFLxNTHhJXu1c  
Vertcoin: VbKfmr7B352WEPs3Qi7VeC4WviGS1jGQvd  


## Sources
U-Net: Convolutional Networks for Biomedical Image Segmentation   
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/   
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.   

ISBI 2012 Segmentation Challenge   
http://brainiac2.mit.edu/isbi_challenge/home   
Ignacio Arganda-Carreras, Srinivas C. Turaga, Daniel R. Berger, Dan Ciresan, Alessandro Giusti, Luca M. Gambardella, Jürgen Schmidhuber, Dmtry Laptev, Sarversh Dwivedi, Joachim M. Buhmann, Ting Liu, Mojtaba Seyedhosseini, Tolga Tasdizen, Lee Kamentsky, Radim Burget, Vaclav Uher, Xiao Tan, Chanming Sun, Tuan D. Pham, Eran Bas, Mustafa G. Uzunbas, Albert Cardona, Johannes Schindelin, and H. Sebastian Seung. Crowdsourcing the creation of image segmentation algorithms for connectomics. Frontiers in Neuroanatomy, vol. 9, no. 142, 2015.   
