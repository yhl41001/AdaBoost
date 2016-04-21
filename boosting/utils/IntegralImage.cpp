/*
 * IntegralImage.cpp
 *
 *  Created on: 21/mar/2016
 *      Author: lorenzocioni
 */

#include "IntegralImage.h"
#include <iostream>

using namespace std;
using namespace cv;

double IntegralImage::computeArea(Mat intImg, Rect r){
	double a1 = intImg.at<double>(r.y + r.height, r.x + r.width);
	double a2 = intImg.at<double>(r.y, r.x);
	double a3 = intImg.at<double>(r.y, r.x + r.width);
	double a4 = intImg.at<double>(r.y + r.height, r.x);
	return a1 + a2 - a3 - a4;
}

Mat IntegralImage::computeIntegralImage(Mat img){
	Mat output;
	integral(img, output, CV_64F);
	return output;
}


