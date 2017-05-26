#Code for image partition using absolute color reduction.
from skimage import io, feature, img_as_ubyte
from color_histogram.core.hist_3d import Hist3D
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

def read_image(path):
	return io.imread(path)

def reduce_dim(item):
	return [i[:-1] for i in item]

def compute_hist(img):
	return Hist3D(img, num_bins=256, color_space='rgb')

def mean_shift(hist, h, w, cs):
	#Mean shift algorithm
	bwd = estimate_bandwidth(hist.rgbColors(), quantile=0.1, n_samples=6000)
	ms = MeanShift(bandwidth=bwd, bin_seeding=True)
	ms.fit(hist.rgbColors())
	
	#Calculating euclidean distance between cluster center and pixels.
	pixels,rgb_pixels = hist.getPixels()
	ed = euclidean_distances(rgb_pixels,ms.cluster_centers_)

	#Extracting closest center for each pixel.
	pix_center = np.argmin(ed,axis=1)

	#Create Image per cluster
	list_of_images = []
	for i in np.unique(ms.labels_):
		temp = pix_center == i
		img_temp = np.ones((h*w,cs))
		for index in enumerate(temp):
			if index[1]:
				img_temp[index[0]] = rgb_pixels[index[0]]
		img_temp = img_temp.reshape((h,w,cs))
		img_temp = img_as_ubyte(img_temp)
		list_of_images.append(img_temp)

	return list_of_images

if __name__ == '__main__':
	img = io.imread("/home/shailesh/word_6.png")

	h,w,cs = img.shape
	
	#delete alpha factors from pixels
	img = np.array([reduce_dim(k) for k in img])

	#Compute color histogram.
	hist = compute_hist(img)
	
	#Apply mean shift algorithm.
	list_of_images = mean_shift(hist, h, w, cs-1)

	io.imshow(list_of_images[1])
	io.show()
	