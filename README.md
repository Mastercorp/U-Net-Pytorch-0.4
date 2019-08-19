[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1da555f619cb4b63915efb74e2c64f4f)](https://www.codacy.com/app/Mastercorp/U-Net-Pytorch-0.4?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Mastercorp/U-Net-Pytorch-0.4&amp;utm_campaign=Badge_Grade)

# Introduction
U-Net-Pytorch-0.4 is a custom U-Net implementation in python 2.7 for Pytorch 0.41.
Furthermore, a custom dataloader is introduced, which can load the ISBI 2012 Dataset.
Dataaugmentation is applied in the dataloader.

Details about the U-Net network can be found on the U-Net [project page](<https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>).
The implementation in this repository is tested on Ubuntu 16.04 with Python 2.7    


## License
The implementation is freely available under the MIT License,
meaning that you can basically do with the code whatever you want.

## Dependencies
*   Pytorch 0.4, Python 2.7, ( CUDA 9.1 for GPU acceleration ) scipy ( for data augmentation ), sacred ( <https://github.com/IDSIA/sacred> )
 
## Building
No building is required, just clone or download the github project in a directory. The programm is tested on Ubuntu 16.04 with a Geforce GTX 1070 8GB Nvidia Driver 390.48 CUDA 9.1, Python 2.7.14 and Pytorch 0.4  

## Usage
/python should start python 2.7 . You can check the version with python --version
```
usage: /python main.py 
```

## Settings 
Sacred is a tool to help you configure, organize, log and reproduce experiments. All important settings can be changed in the config.json file. ( only the dataset direction is hardcoded into main.py at line 176 to 187. If you use another dataset, just change the used direction. )

*   `batch_size`   mini batch size (default: 1). For 8k memory on gpu, minibatchsize of 2-3 possible for ISBI 2012
*   `workers`     number of data loading workers (default: 2)
*   `learningrate`                initial learning rate (default: 0.001)
*   `momentum`          momentum (default: 0.99)
*   `weightdecay`        weight decay (L2 penalty ) (default:0)
*   `epochs`            number of total epochs to run (default: 600)
*   `resume`      relative path to latest checkpoint, load all needed data to resume the network (default: none)
*   `evaluate`        evaluate model on validation set
*   `saveimages`     save the first image of output each epoche
*   `cpu`             use cpu instead of gpu
*   `padding`             use padding at each 3x3 convolution to maintain image size
*   `txtinfo`                  save console output in txt
*   `classweight`                 use classweights

## Examples
Use the ISBI2012 dataset and run for 600 epochs. Use padding at each 3x3 convolution and save information about the used settings and losses each epoch in a txt file.
```
python main.py
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
Bitcoin: 17C7sAiGw7a7g7wZx3eVbe5Vuqs35EJhSL  
Litecoin: LLCmNPWBt8TxnNuaTWfyDdajXM5rZPuzsT  
Verge   : DN5QsxVaFLykVykGNFqRdGFvi5zg2zz3Rq   
Reddcoin: Rsjofwt2TNu6Gf6eQdzyUhJe2J6vLsKoBq  
Dogecoin: DM13fiivexaZf35HTQ7AKBFLxNTHhJXu1c  
Vertcoin: VbKfmr7B352WEPs3Qi7VeC4WviGS1jGQvd  

## Sources
U-Net: Convolutional Networks for Biomedical Image Segmentation   
<https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/> 
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.   

ISBI 2012 Segmentation Challenge   
<http://brainiac2.mit.edu/isbi_challenge/home>   
Ignacio Arganda-Carreras, Srinivas C. Turaga, Daniel R. Berger, Dan Ciresan, Alessandro Giusti, Luca M. Gambardella, Jürgen Schmidhuber, Dmtry Laptev, Sarversh Dwivedi, Joachim M. Buhmann, Ting Liu, Mojtaba Seyedhosseini, Tolga Tasdizen, Lee Kamentsky, Radim Burget, Vaclav Uher, Xiao Tan, Chanming Sun, Tuan D. Pham, Eran Bas, Mustafa G. Uzunbas, Albert Cardona, Johannes Schindelin, and H. Sebastian Seung. Crowdsourcing the creation of image segmentation algorithms for connectomics. Frontiers in Neuroanatomy, vol. 9, no. 142, 2015.   
