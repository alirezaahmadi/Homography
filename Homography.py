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

import cv2 as cv
import numpy as np
from cv2 import xfeatures2d, Feature2D 
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def main():
    """Homography of Depth/IR image to an RGB image usinf Keypoint detection
    """
    parser = argparse.ArgumentParser(description="Register a Depth/IR image to an RGB image")
    parser.add_argument("target_img", help="target image address *.png")
    parser.add_argument("source_img", help="source image address *.png")
    parser.add_argument("output_dir", help="Output directory to save  registered image.")

    args = parser.parse_args()

    print("Source image:" ,args.source_img, "will be registered to ", args.target_img, "...")

    sourceImage = cv.imread(args.source_img)        
    targetImage = cv.imread(args.target_img) 

    target_gray= cv.cvtColor(targetImage, cv.COLOR_BGR2GRAY)
    source_gray= cv.cvtColor(sourceImage, cv.COLOR_BGR2GRAY)

    result = homography(target_gray, source_gray, minMatchPoint = MIN_MATCH_COUNT) 

    cv.imwrite(args.output_dir, result)     

    return

def homography(targetImage, sourceImage, minMatchPoint):
    
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(targetImage,None)
    kp2, des2 = sift.detectAndCompute(sourceImage,None)

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

    img=cv.drawKeypoints(targetImage,kp1,targetImage,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('sift_keypoints',img)
    cv.waitKey()
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = targetImage.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        sourceImage = cv.polylines(sourceImage,[np.int32(dst)],True,255,3, cv.LINE_AA)
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv.drawMatches(targetImage,kp1,sourceImage,kp2,good,None,**draw_params)
        plt.imshow(img3, 'gray'),plt.show()

        warp = cv.warpPerspective(sourceImage, M, (sourceImage.shape[1], sourceImage.shape[0]), flags=cv.WARP_INVERSE_MAP)

        alpha = 0.5
        beta = (1.0 - alpha)
        dst = cv.addWeighted(targetImage, alpha, warp, beta, 0.0)
        cv.imshow('dst', dst)
        cv.waitKey()
        cv.destroyAllWindows()

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    
    return dst

   
if __name__ == "__main__":
    main()
