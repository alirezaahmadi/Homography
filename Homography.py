#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;

    Mat rgb = imread("../frame_5.png", IMREAD_GRAYSCALE);
    Mat ir = imread("../frame_5a.png", IMREAD_GRAYSCALE);


    if(!rgb.data || rgb.empty())
    {
        cerr << "Problem loading rgb image!!!" << endl;
        return -1;
    }

    if(!ir.data || ir.empty())
    {
        cerr << "Problem loading ir image!!!" << endl;
        return -1;
    }

    Mat ir2 = ir * 2;


    imshow("rgb", rgb);
    imshow("ir", ir);
    imshow("ir2", ir2);


    waitKey();




    //-- Step 1: Detect the keypoints and extract descriptors using SURF
    int minHessian = 400;

    Ptr<SURF> detector = SURF::create( /*minHessian*/600);
//    Ptr<BRISK> detector = BRISK::create(100);


    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( rgb, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( ir, Mat(), keypoints_scene, descriptors_scene );

    //-- Step 2: Matching descriptor vectors using FLANN matcher
    BFMatcher matcher(NORM_L2, true);
//    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
//    for( int i = 0; i < descriptors_object.rows; i++ )
    for( int i = 0; i < matches.size(); i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
//    for( int i = 0; i < descriptors_object.rows; i++ )
    for( int i = 0; i < matches.size(); i++ )
    { if( matches[i].distance <= 0.3/*2*min_dist*/ )
       { good_matches.push_back( matches[i]); cout << matches[i].distance << endl; }
    }
    Mat img_matches;
    drawMatches( rgb, keypoints_object, ir, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    cout << obj.size() << " :: " << scene.size() << endl;

    // give manually the matching points (TODO: use a mousecallback input instead)
    vector<Point2f> manual_obj = {Point2f(124,91), Point2f(300,90), Point2f(311,212), Point2f(305,349), Point2f(112,346), Point2f(111,211), Point2f(107,91), Point2f(326,104), Point2f(332,202), Point2f(364,191), Point2f(331,318), Point2f(96,316), Point2f(94,200), Point2f(61,182), Point2f(360,86), Point2f(363,128), Point2f(385,209), Point2f(395,252), Point2f(357,347), Point2f(65,335), Point2f(56,288), Point2f(21,243), Point2f(39,203), Point2f(31,157), Point2f(40,120), Point2f(354,128), Point2f(56,355), Point2f(69,355), Point2f(60,178), Point2f(68,120), Point2f(345,86), Point2f(75,91), Point2f(311,202), Point2f(309,318), Point2f(111,316), Point2f(110,200), Point2f(352,376), Point2f(212,210), Point2f(212,85), Point2f(212,352), Point2f(392,303), Point2f(394,337)};


    vector<Point2f> manual_scene = {Point2f(46,11), Point2f(109,11), Point2f(111,55), Point2f(111,104), Point2f(42,103), Point2f(43,55), Point2f(38,9), Point2f(120,16), Point2f(119,52), Point2f(132,48), Point2f(120,93), Point2f(37,93), Point2f(36,51), Point2f(23,45), Point2f(134,6), Point2f(134,24), Point2f(141,54), Point2f(145,70), Point2f(132,105), Point2f(23,102), Point2f(20,84), Point2f(6,67), Point2f(13,52), Point2f(8,34), Point2f(11,20), Point2f(130,24), Point2f(17,110), Point2f(23,110), Point2f(22,43), Point2f(23,20), Point2f(127,6), Point2f(25,9), Point2f(112,52), Point2f(110,93), Point2f(42,93), Point2f(42,51), Point2f(131,119), Point2f(77,56), Point2f(77,10), Point2f(77,105), Point2f(145,90), Point2f(148,103)};


    Mat crgb, cir;
    cvtColor(rgb,crgb, CV_GRAY2BGR);
    cvtColor(ir,cir, CV_GRAY2BGR);

    for(size_t i = 0; i < manual_obj.size(); i++)
    {
        circle(crgb, manual_obj[i], 2, CV_RGB(255,0,0), -1);
    }
    for(size_t i = 0; i < manual_scene.size(); i++)
    {
        circle(cir, manual_scene[i], 2, CV_RGB(255,0,0), -1);
    }

    imshow("circles", crgb);
    imshow("circles_ir", cir);
    waitKey();

    Mat H = findHomography( manual_scene, manual_obj, 0);
    cout << "H: " << H << endl;

    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );

    Mat registration;

    warpPerspective(ir, registration, H, rgb.size(), CV_INTER_CUBIC);


    imshow("ir2rgb", registration);


    Mat addweight;

    addWeighted( rgb, 0.3, registration, 0.9, 0.0, addweight); // blend src image with canny image

    imshow("addwweighted", addweight );

    waitKey();

    cout << "Goobye World!" << endl;
    return 0;
}
