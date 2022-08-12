import numpy as np
import os
import random
from PIL import Image, ImageDraw
import cv2
import uuid
import glob

# Use locations of item along with their values to determine colors. I could then put them as unique colors at random locations.

# Maybe, for each subjects image each pixel location corresponds to a particular question and each color for each pixel corresponds
# to the answer.

# Or perhaps make some of the encoding/decoding outside the image e.g. each pixel location correpsonds to a prime number (so each location
# corresponds to a question) then each pixel color value corresponds to that prime number to the power of the answer. This will only
# work for images containing the number of questions less than the fifth root of 256^3 (for 1-5 answers). This will give a unique Godel number. Looks like
# I can get 27.85, so 27 questions out of this. Then making it a square image, I can do a 5x5 = 25 question image.
# - Seems that there ar problems with this, first as the 27th prime number is 103 and 103^5 is greater than 256^3, I had forgotten that
# there are gaps between prime numbers.
# I could instead just make the pixel locations the questions and make their colors not unique but all following a pattern of being
# one of 5 colors (one color for each of the answers). Or maybe make more colors by multiplying each of the 5 potential color values
# by pixel locations.
# We could also build a unique number by stacking each of the answers. So for example question 1,2,and 3 have a value of 3,5,4 respectively
# which would give the number 354 and as long as the questions only have values 0-9 we have a unique number. Now I could just give a color
# coding simply based on the answer or I could multiply the answer by the location and convert that in to RGB values. Perhaps I could
# also factor in the sum of scores on the set of questions as a sort of hash check for each image. That is, each of the images has a color
# coding decided in part by the overall sum of scores.

# Now to create a basic image from a column of questions with just 5 potential colors.

# for each column, need to translate that to a pixel value


def multiply_pixel(n):
    # Need to make maxval dependent on log or something.
    m = n * 30
    # m = pow(n,NumberofBins)
    return m

def higher_power(n):
    m = pow(n,3)
    return m

def Create_RGB_tuples(List_In):
    res = [(val,val,val) for val in List_In]
    return res

def LabelImage(BinnedScore):
    Binned_List = []
    [Binned_List.append([i]) for i in BinnedScore]
    return Binned_List

def write_greyscaleImage_toFile(subject, Image,width, height, ImagePath):
    # Write image to file
    Image_filename = "Image_" + str(subject)
    # The_image = Image.frombytes('L', (width, height), Image)

    Image.save(ImagePath + "/{Image_file_name}.png".format(Image_file_name=Image_filename))

def load_images(df, inputPath):
	images = []
	# loop over the indexes of the houses
	for i in df.index.values:
		# find the four images for the house and sort the file paths,
		# ensuring the four are always in the *same order*
		basePath = os.path.sep.join([inputPath, "Image_"+str(i)+".png"])
		housePaths = sorted(list(glob.glob(basePath)))

		# initialize our list of input images along with the output image
		# after *combining* the four input images
		inputImages = []
		outputImage = np.zeros((64, 64, 3), dtype="uint8")

		# loop over the input house paths
		for housePath in housePaths:
			# load the input image, resize it to be 32 32, and then
			# update the list of input images
			image = cv2.imread(housePath)
			image = cv2.resize(image, (32, 32))
			inputImages.append(image)

def Questionnaire_Image(width=5, height=5, Item_vals=0):
    from SupportFunctions import merge_Lists
    # Apply the function higher_power to all elements in the Item_vals list.
    empowered_list = list(map(multiply_pixel, Item_vals))
    # Create a list as long as the number of pixels in the new image.
    blank_image_list = [255 for _ in range(width * height)]
    # Merge the list (of questions) with that of the list for image pixels
    Q_Image_list = merge_Lists(blank_image_list,empowered_list)
    Tuples = Q_Image_list
    Outputimagesize = (width, height)
    dst_image = Image.new('L', Outputimagesize)
    dst_image.putdata(Tuples)
    return dst_image