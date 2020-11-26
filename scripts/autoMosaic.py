#!/usr/bin/env python
"""
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Homography of Depth/IR image to an                        %
% RGB image usinf Keypoint detection                        %   
% by: Alireza Ahmadi,                                       %
% University of Bonn- AI & Robotics Researcher              %
% Alireza.Ahmadi@uni-bonn.de                                %
% AlirezaAhmadi.xyz                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
"""

import os
import argparse
import glob
import cv2 as cv
import numpy as np
from cv2 import Feature2D 
from matplotlib import pyplot as plt

import homography

MIN_MATCH_COUNT = 100

def argParser():
    parser = argparse.ArgumentParser(description="Register a Depth/IR image to an RGB image")
    parser.add_argument("dataset_dir", help="address of folder contaiing .png images")
    parser.add_argument("output_dir",  help="Output directory to save  registered image.")
    parser.add_argument("num",  type=int, help="number of images to warp.")
    # parser.add_argument("device", help="GPU or CPU.")
    return parser.parse_args()

def main():
    """Homography of Depth/IR image to an RGB image usinf Keypoint detection
    """
    args = argParser()
    
        
    result =[]
    imagesList = sorted( glob.glob( os.path.join( args.dataset_dir, '*.' + 'png' ) ) )
    print("Number of Loaded Images: ", len(imagesList))

    numOfImages = args.num
    if numOfImages == 0:
        numOfImages = len(imagesList)

    targetImage = cv.imread(imagesList[0]) 
    

    for imageIndex in range(1,numOfImages):
        print("frameNum: ", imageIndex)
        sourceImage_adr = imagesList[imageIndex]
        sourceImage = cv.imread(sourceImage_adr) 

        source_gray= cv.cvtColor(sourceImage, cv.COLOR_BGR2GRAY)
        target_gray= cv.cvtColor(targetImage, cv.COLOR_BGR2GRAY)

        M = homography.homography(target_gray, source_gray, minMatchPoint = MIN_MATCH_COUNT) 
        result = homography.warpTwoImages(sourceImage, targetImage, M)
        cv.namedWindow('image',cv.WINDOW_NORMAL)
        cv.resizeWindow('image', 600,600)
        cv.imshow('image', result) 
        cv.waitKey(1)

        targetImage = result.copy()    
    
    cv.imshow(args.output_dir, result)  
    cv.waitKey()
    cv.destroyAllWindows()
    return

    # else:
    #     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    #     matchesMask = None
    
    # return dst
  
if __name__ == "__main__":
    main()