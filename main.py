import os
import glob
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from ImageCrop import ImageCrop


RESIZE = (256, 256)


# http://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir-python
def _listdir(path):
	return glob.glob(os.path.join(path, '*'))


def stackImages(filenames):
	numExamples = len(filenames)
	numFeatures = RESIZE[0] * RESIZE[1]
	stacked = np.zeros((numExamples, numFeatures))
	for i in range(numExamples):
		currImage = cv2.imread(filenames[i], cv2.IMREAD_GRAYSCALE)
		stacked[i, :] = currImage.flatten()

	return stacked


def __main__():
	pwd = os.getcwd()
	buttonFolder = pwd + '/Buttons/sorted/'
	saveFolder = pwd + '/Buttons/saved/'
	resizeFolder = pwd + '/Buttons/resized/'

	filenames = _listdir(buttonFolder)
	for filename in filenames:
		print "Current file: ", filename
		imageCrop = ImageCrop(filename)

		print "Amount of circles detected: ", imageCrop.buttonAmount

	# Rename: numeric, resize to 256x256
	filenames = _listdir(saveFolder)
	numExamples = len(filenames)
	for i in range(numExamples):
		fileRename = saveFolder + str(i + 1) + '.png'
		resizeName = resizeFolder + str(i + 1) + '.png'
		
		image = plt.imread(filenames[i])
		resizedImage = cv2.resize(image, RESIZE)
		plt.imsave(resizeName, resizedImage)

		os.rename(filenames[i], fileRename)

	#stacked = stackImages(_listdir(resizeFolder))


if __name__ == '__main__':
	sys.exit(__main__())