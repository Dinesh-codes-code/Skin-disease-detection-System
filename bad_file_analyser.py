
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:06:17 2021

@author: My Pc
"""
import os
from os import listdir
from PIL import Image

dir_path = "F:/ac/Actinic Keratosis"


for filename in listdir(dir_path):
    if filename.endswith('.jpg'):
        try:
            img = Image.open(dir_path+"\\"+filename) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename)