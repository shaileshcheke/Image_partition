# Code used sobel operator  to extract gradient magni and direction.

from skimage import io, feature, img_as_ubyte, filters
from skimage.color import rgb2gray
import numpy as np
if __name__ == "__main__":
	img = io.imread(str(raw_input("Please provide path of image\n")))
	h,w,cs = img.shape
	img_g = rgb2gray(img)

	#apply horizontal and vertical sobel operator to
	#extract gradient in vertical and horizontal direction.
	img_v = filters.sobel_v(img_g)
	img_h = filters.sobel_h(img_g)
	
	#gradient mag can be directly calculated.
	img_m = filters.sobel(img_g)

	#Direction calculation for gradient.
	np.degrees(np.arctan2(img_v, img_h))

	