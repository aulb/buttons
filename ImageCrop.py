import cv2
import numpy as np 
from matplotlib import pyplot as plt
import cv2.cv as cv 
import math
import json
import os

# Helper function
# http://stackoverflow.com/questions/15072736/extracting-a-region-from-an-image-using-slicing-in-python-opencv/15074748
def bgr2rgb(image):
	b = image[:,:,0]
	g = image[:,:,1]
	r = image[:,:,2]
	return cv2.merge([r,g,b])


#http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
def _unique(a):
	order = np.lexsort(a.T)
	a = a[order]
	diff = np.diff(a, axis=0)
	ui = np.ones(len(a), 'bool')
	ui[1:] = (diff != 0).any(axis=1) 
	return a[ui]


# Temporary, get bounding box from circle 
def getBoundingBox(circle, scale=1):
	x1,y1 = circle[0:2] - circle[2]
	x2,y2 = circle[0:2] + circle[2]

	if scale:
		x1,y1,x2,y2 = np.array([x1,y1,x2,y2]) * scale

	# use as image[y1:y2, x1:x2]
	return x1,y1,x2,y2


def preprocess(channel, ratio=1):	
	# Minimum height, width
	width, height = channel.shape
	if ratio is not 1:
		channel = cv2.resize(channel, (0,0), fx=ratio, fy=ratio)

	channel = cv2.medianBlur(channel, 5)
	channel = cv2.GaussianBlur(channel, (5,5), 0)
	return channel


def cropImage(image, indx):
	x1,y1,x2,y2 = indx	

	if len(image.shape) > 2:
		return image[y1:y2, x1:x2, :]
	else:
		return image[y1:y2, x1:x2]


def getBoundingBox(circle, scale=1):
	x1,y1 = circle[0:2] - circle[2]
	x2,y2 = circle[0:2] + circle[2]

	if scale:
		x1,y1,x2,y2 = np.array([x1,y1,x2,y2]) * scale

	return x1,y1,x2,y2


def whichCircle(img, circle):
	coord = getBoundingBox(circle, 1)
	plt.imshow(cropImage(img, coord))
	plt.show()


def showIm(img):
	plt.imshow(img)
	plt.show()


class ImageCrop():
	"""Deals with cropping buttons in an image. Images contains
	a specific sized buttons and type.

	Attributes:
		buttonAmount	: The amount of buttons in this image

	"""


	CONFIGFILE = 'ImageCrop.json'


	def __init__(self, filename):
		# If index is provided then use that instead of the whole image 
		self.filename = filename # Filename follows a specific format

		try:
			self.buttonSize = filename.split('/')[-1].split('_')[0]
		except Exception as e:
			print("Need to name the images properly.")
			raise

		# Load the images
		self.loadImage()
		self.loadParameters()

		# Create save folders if not defined
		if not os.path.isdir(self.saveDestination):
			os.makedirs(self.saveDestination)

		self.executeCropping()


	def loadImage(self):
		# Flip opencv BGR to RGB
		imageGray = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
		imageRGB = bgr2rgb(cv2.imread(self.filename, cv2.IMREAD_COLOR)) 

		if imageGray is None or imageRGB is None:
			raise cv2.error('Image not loaded. Corrupted or missing file.') 

		self.imageGray = preprocess(imageGray)
		self.imageRGB = imageRGB
		height, width = imageGray.shape
		self.height = height
		self.width = width


	# Load parameters
	def loadParameters(self):
		# Hardcoded ImageCrop.json
		with open(self.CONFIGFILE) as conf:
			data = json.load(conf)

		commonParams = data["commonParams"]
		sizeParams = data["sizeParams"][self.buttonSize]

		self.dp = commonParams["dp"]
		self.method = commonParams["method"]
		self.mSigma = commonParams["mSigma"]
		self.cannyThresh = commonParams["cannyThresh"]
		self.linesThresh = commonParams["linesThresh"]
		self.minDistRatio = commonParams["minDistRatio"]
		self.saveDestination = commonParams["saveDestination"]

		self.minRadius = sizeParams["minRadius"]
		self.maxRadius = sizeParams["maxRadius"]
		self.minDist = sizeParams["minDist"]


	def executeCropping(self):
		if self.buttonSize not in 'SNMLX':
			raise Exception('Unknown size.')

		# Get and merge detected circles
		self.getCircles()
		self.mergeCircles()

		if self.circles is not None:
			# Crop the images into buttons
			self.cropButtons()


	def cropButtons(self):
		# Determine name
		counter = 1
		baseFilename = self.filename.split('/')[-1].split('.')[0] + '_'

		for circle in self.circles:
			boxCoord = getBoundingBox(circle)
			button = cropImage(self.imageRGB, boxCoord)
			buttonFilename = self.saveDestination + baseFilename + \
							str(counter) + '.png'

			plt.imsave(buttonFilename, button)
			counter += 1

		self.buttonAmount = counter - 1


	# Wrapper for cv2.HoughCircles to do it a couple times
	def getCircles(self):
		counter = 0 
		modes = ['expr', 'otsu', 'meanmed'] # Hardcoded in
		circles = None	
		# Initialize circles
		while circles is None and counter < len(modes):
			currParams = self.createCircleParams(modes[counter])
			circles = cv2.HoughCircles(**currParams) # make sure not None
			counter += 1

		# Stack circles
		while counter < len(modes):
			currParams = self.createCircleParams(modes[counter])
			newCircles = cv2.HoughCircles(**currParams)
			if newCircles is not None:
				circles = np.hstack((circles, newCircles))

			counter += 1

		self.circles = circles
		if circles is not None:
			# Make into int type
			# Shave off the weird shape (1, x, 3) from opencv
			uniqueCircles = _unique(np.around(circles).astype(int)[0])
			self.circles = uniqueCircles


	def mergeCircles(self):
		oldCircles = self.circles

		# Initial amount of circles detected and average radius
		numDetectedCircles = oldCircles.shape[0]
		avgCircleRadius = np.mean(oldCircles[...,2])
		# Minimum distance before grouping them together
		minDist = self.minRadius * self.minDistRatio # HYPERPARAMETER

		circlesDict = {}
		distMatrix = np.zeros((numDetectedCircles, numDetectedCircles))
		uniqueCircleCounter = 0
		grouped = []

		# Create the upper triangular of "distance matrix"
		for i in range(numDetectedCircles):
			for j in range(i + 1, numDetectedCircles):
				distance = (oldCircles[i, 0:2] - oldCircles[j, 0:2])
				distMatrix[i, j] = np.linalg.norm(distance)

		# Starts grouping here
		for i in range(numDetectedCircles):
			# If the circle is already grouped
			if i in grouped:
				continue

			initialized = False
			for j in range(i + 1, numDetectedCircles):
				if distMatrix[i, j] <= minDist:
					if not initialized:
						uniqueCircleCounter += 1
						initialized = True
						circlesDict[uniqueCircleCounter] = [i, j]
						grouped.append(i)
					else:
						circlesDict[uniqueCircleCounter].append(j)
					grouped.append(j)

			if not initialized:
				uniqueCircleCounter += 1
				circlesDict[uniqueCircleCounter] = [i]

		# Merge circles by taking the biggest one, or if the
		# biggest is less than mean then expand the circle
		mergedCircles = np.zeros((len(circlesDict), 3))
		for key in circlesDict:
			currCircles = oldCircles[circlesDict[key]]
			currSortedIdx = np.argsort(currCircles[...,2])
			currCircles = currCircles[currSortedIdx[::-1]]
			currBiggestRadius = currCircles[0,2]
			
			mergedCircles[key - 1] = currCircles[0]

			if currBiggestRadius < avgCircleRadius:
				# if its not bigger than the average, expand
				# do average mean, create new center
				# create new radius that is a little bigger

				radiusWeights = currCircles[:,2] / float(np.sum(currCircles[:,2])) 
				if currCircles.shape[0] is not 1:
					newX = int(np.dot(currCircles[:,0], radiusWeights))
					newY = int(np.dot(currCircles[:,1], radiusWeights))
					newR = int(math.floor((avgCircleRadius + currBiggestRadius) / 2.))

					mergedCircles[key - 1, 0] = newX
					mergedCircles[key - 1, 1] = newY
					mergedCircles[key - 1, 2] = newR

		mergedCircles = mergedCircles.astype(int)
		self.circles = mergedCircles


	def createCircleParams(self, paramMethod='expr'):
		# Determine the rest of the parameter
		if paramMethod == 'otsu':
			otsu = int(math.floor(
				cv2.threshold(self.imageGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]))
			param1 = otsu
			param2 = otsu
		elif paramMethod == 'meanmed':
			mean = min(255, int(math.floor(np.mean(self.imageGray) * (1.0 + self.mSigma))))
			median = min(255, int(math.floor(np.median(self.imageGray) * (1.0 + self.mSigma))))
			param1 = min(mean, median)
			param2 = max(mean, median)
		else:
			param1 = self.cannyThresh
			param2 = self.linesThresh

		circleParams = {
			"image": self.imageGray,
			"method": cv.CV_HOUGH_GRADIENT,
			"dp": self.dp,
			"minDist": self.minDist, # The minimum distance between circle detected
			"param1": param1, # Canny edge detection
			"param2": param2, 
			"minRadius": self.minRadius,
			"maxRadius": self.maxRadius		
		}

		return circleParams