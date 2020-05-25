# Homography
A python implementation of homography algorithm by SIFT descriptors  


## Dependencies
install opencv and opencv-contribLib using 

pip install opencv-python opencv-contrib-python-nonfree

## Example Command o run

python3 Homography.py images/target.png images/source.png images/homography_result.png

--- 
**To get Help option **

python Homography.py -h

usage: Homography.py [-h] target_img source_img output_dir

Register a Depth/IR image to an RGB image

positional arguments:
  target_img  target image address *.png
  source_img  source image address *.png
  output_dir  Output directory to save registered image.

optional arguments:
  -h, --help  show this help message and exit
  
  
