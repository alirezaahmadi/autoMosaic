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

import cv2 as cv
import numpy as np
from cv2 import Feature2D 
from matplotlib import pyplot as plt

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    result = cv.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

    # Apply panorama correction
    # width = img2.shape[1] + img1.shape[1]
    # height = img2.shape[0] + img1.shape[0]

    # result = cv.warpPerspective(img2, H, (width, height))
    # result[0:img1.shape[0], 0:img1.shape[1]] = img1
    return result



def homography(targetImage, sourceImage, minMatchPoint):
    
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(targetImage,None)
    kp2, des2 = sift.detectAndCompute(sourceImage,None)
    result = []
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    # print("Num of Pg:", len(good))
    # img=cv.drawKeypoints(targetImage,kp1,targetImage,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow('sift_keypoints',img)
    
    if len(good)>minMatchPoint:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,100.0)

        
        # matchesMask = mask.ravel().tolist()
        # h,w = targetImage.shape
        # pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv.perspectiveTransform(pts,M)
        # sourceImage = cv.polylines(sourceImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #            singlePointColor = None,
        #            matchesMask = matchesMask, # draw only inliers
        #            flags = 2)
        # img3 = cv.drawMatches(targetImage,kp1,sourceImage,kp2,good,None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()
        # cv.waitKey()

        # warp = cv.warpPerspective(sourceImage, M, (sourceImage.shape[1], sourceImage.shape[0]), flags=cv.WARP_INVERSE_MAP)

        # result = cv2.warpPerspective(sourceImage, Ht.dot(M), (xmax-xmin, ymax-ymin))
        # result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        # cv.imshow('dst', result)
        # cv.waitKey()
        # cv.destroyAllWindows()

    return M