
#Import required libraries
import cv2
from glob import glob
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageStat
import numpy as np
import pandas as pd
from random import random
  
#CREATE MAPPINGS BETWEEN THE RANGE OF BRIGHTNESS VALUE OF PIXELS 
# mapped = []
# d = {range(0, 25): 1, range(26, 50): 2, range(51, 75): 3, range(76,100): 4, range(101,125): 5, range(126,150): 6, range(151, 175): 7 
#     , range(176, 200): 8, range(201,225): 9, range(226,255): 10}

# lst = ['1', '2', '3', '4', '5','6','7','8','9','10']
# mapped = list(map(d.get,lst))
# print(mapped)

test_im= " GIVE YOUR PATH "  #test image input path
im_file = " GIVE YOUR PATH "  #CHANGE PATH OF FILE AS REQUIRED ONE

random.shuffle(im_file)


#ALTERNATE METHOD..

# #Read image, Calculate Global-Statistic value of the image , Calculate average value of the three bands and returns it 
# def brightness(im_file):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    r,g,b = stat.mean
#    return math.sqrt(0.299*(r**2) + 0.587*(g**2) + 0.114*(b**2))

# avg = avg_brightness(test_im)
# print('Avg brightness: ' + str(avg))
# plt.imshow(test_im)

def resize(rgb_image):

  height=512 
  width=512
  img = cv2.resize(rgb_image,(height,width))

  return img


#DISPLAY THE IMAGES IN HUE,SATURATION AND VALUE CHANNELS

def disp_artificial_lights(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    cols = 2
    rows = 2
    
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    fig = plt.figure(figsize=(20,10))
    
    ax = fig.add_subplot(rows, cols, 1)
    ax.set_title("Original")
    ax.imshow(rgb_image)

    ax = fig.add_subplot(rows, cols, 2)
    ax.set_title("H {}".format(np.mean(h)))
    ax.imshow(h,cmap='gray', vmin=0, vmax=255)
    
    ax = fig.add_subplot(rows, cols, 3)
    ax.set_title("S")
    ax.imshow(s,cmap='gray', vmin=0, vmax=255)

    ax = fig.add_subplot(rows, cols, 4)
    ax.set_title("V")
    ax.imshow(v,cmap='gray', vmin=0, vmax=255)
    
    plt.show()


# Find the average Value or brightness of an image
def avg_brightness(rgb_image):
    
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    

    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:,:,2])
    area = rgb_image.shape[0]*rgb_image.shape[1]  # pixels
    
    # find the avg
    avg = sum_brightness/area
    
    return avg


thresh = 80
thresh1= 180

# This function should take in RGB image input
def estimator(rgb_image):
    
    # Extract average brightness feature from an RGB image 
    avg = avg_brightness(rgb_image)
        
    # Use the avg brightness feature to predict a label (0, 1)
    label = 0   

    if(avg > thresh1):
      label = 1  #More brighter image
   
    if(avg>thresh && avg<thresh1):
      label = 0.5

    return label
    

def get_images(test_images):

    for image in im_file:

       disp_artificial_lights(image)

       x = resize(image)

       p_label = estimator(x)








