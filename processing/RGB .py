# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:08:59 2020

@author: ion-m
"""


import cv2
import numpy as np
img = cv2.imread('data800/weak damage/13.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = np.zeros_like(img)
img2[:,:,0] = gray
img2[:,:,1] = gray
img2[:,:,2] = gray
cv2.imwrite('data800/weak damage/13.jpg', img2)