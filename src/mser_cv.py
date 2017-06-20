# Maximal Stable Extrimal Region.
from skimage import io, feature, img_as_ubyte
from skimage.color import rgb2gray
import numpy as np
import cv2

if __name__ == '__main__':

	img = io.imread(raw_input('Please provide path of image\n'))

	# Get gray scale version of image.
	gray = img_as_ubyte(rgb2gray(img))

	# Create mser object
	mser = cv2.MSER()

	# Detect mser regions
	regions = mser.detect(gray)

	# Create polygonal boundy boxes aroun blobs
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	img_c = img.copy()
	cv2.polylines(img_c, hulls, 1, (0, 255, 0))

	# Display Image.
	io.imshow(img_c)
	io.show()