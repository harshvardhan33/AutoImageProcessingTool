# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 00:11:05 2020

@author: harshvardhan
"""

import cv2
import numpy as np
from math import sqrt,exp



def meanFilter(image,ksize):
    new_image = cv2.blur(image,ksize)
    return new_image
    

def medianFilter(image,ksize):
    new_image = cv2.medianBlur(image, ksize[0])
    return new_image

def gaussianFilter(image,ksize):
    new_image = cv2.GaussianBlur(image,ksize,15)
    return new_image
    

def laplacianFilter(image,ks):
    new_image = cv2.Laplacian(image,cv2.CV_64F,ksize=ks[0])
    new_image = cv2.convertScaleAbs(new_image)
    return new_image +image


def cannyEdge(image,ksize):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray= cv2.GaussianBlur(grayImage,ksize,0)
    gray= cv2.Canny(gray,100,300)
    return gray

def sobelEdge(image,ks):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(grayImage,cv2.CV_64F,1,0,ksize=ks[0]) 
    sobely = cv2.Sobel(grayImage,cv2.CV_64F,0,1,ksize=ks[0]) 
    new_image = cv2.convertScaleAbs(sobelx+sobely)
    return new_image

def laplacianEdge(image,ks):
    laplacian = cv2.Laplacian(image,cv2.CV_64F,ksize=ks[0]) 
    new_image = cv2.convertScaleAbs(laplacian)
    return new_image

def toRGB(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return new_image

def toYCrCb(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) 
    return new_image

def toHSV(image):
    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return new_img

def toLAB(image):
    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return new_img


def toHeatmap(image):
    new_img= cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return new_img

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base


def brighten(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert image to HSV color space
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*1.25 # scale pixel values up for channel 1
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*1.25 # scale pixel values up for channel 2
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def enhance(img):
    dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    dst2 = cv2.filter2D(img, -1, kernel_sharpening)
    return dst,dst2

def gamma_function(channel, gamma):
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8") #creating lookup table
    channel = cv2.LUT(channel, table)
    return channel

def cartoon(img):
    edges1 = cv2.bitwise_not(cv2.Canny(img, 100, 200)) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5) 
    dst = cv2.edgePreservingFilter(img, flags=2, sigma_s=64, sigma_r=0.25)
    cartoon1 = cv2.bitwise_and(dst, dst, mask=edges1)
    return cartoon1
    
    
def warm(img):
    img[:, :, 0] = gamma_function(img[:, :, 0], 1.25)
    img[:, :, 2] = gamma_function(img[:, :, 2], 0.75)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = gamma_function(hsv[:, :, 1], 0.8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def cold(img):
    img[:, :, 0] = gamma_function(img[:, :, 0], 0.75) 
    img[:, :, 2] = gamma_function(img[:, :, 2], 1.25) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = gamma_function(hsv[:, :, 1], 1.2) 
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img




def sketch(img):
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05) # inbuilt function to generate pencil sketch in both color and grayscale
    return dst_gray,dst_color

def inversion(image):
    res = cv2.bitwise_not(image)
    return res
