# FSC - Fast Scan Cropper

We present FSC - the Fast Scan Cropper - a tool for extracting pictures from scans images with white background.
FSC is written in Python3 and exploiting many builtin function from numpy to achieve quickly cropped images from your scan.
With a clever edge identification, the tool is able to rotate the images if necessary.

FSC was build with regard to the following requirements:

1. Easy and intuitive handling on command line without any major installation
2. Accurate preprocessing of the picture, i.e., appropriate cropping and rotation


### Prerequisites
FSC needs a *Python3* interpreter and the numpy, PIL libraries preinstalled and an image to crop, say:
<p style="text-align: center;"><img src="https://raw.githubusercontent.com/dominik-kopczynski/FSC/master/image/scan_0001.jpg" width="300"/></p>

### Using FSC

To run the script, simply type in:

```
python3 FSC.py your-image.jpg
```
Depending on the number of identified pictures in the image, these pictures will be stored in the same folder with "IMG" prefix and incremental number.

When only a preview is desired, just type in:

```
python3 FSC.py -p your-image.jpg
```
