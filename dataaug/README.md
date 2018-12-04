# Usage

usage: gradaug.py [-h] [-img IMAGEFILE] [-smin SIGMAMIN] [-stepsize INTEGER]
                  [-smax SIGMAMAX] [-iter INTEGER] [-scale INTEGER]


## needed arguments
*  `-img IMAGEFILE`     choose the image file which should be augmented
## Optional arguments
*  `-h, --help`         show the help message
*  `-smin SIGMAMIN`     minimal sigma value for gaussian blur to reduce noise (default:70)
*  `-smax SIGMAMAX`     maximal sigma value for gaussian blur to reduce noise (default:100)
*  `-stepsize INTEGER`  stepsize between sigma values (default:10)
*  `-iter INTEGER`      number of iterations in each sigma setting
*  `-scale INTEGER`     scaling factor which is applied to the gradients  (default:1000)







## Example

python gradaug.py -img test.jpg -smin 70 -smax 100 -stepsize 10 -iter 10 -scale 1000
