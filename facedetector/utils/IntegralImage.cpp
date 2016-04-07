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

IntegralImage::IntegralImage(Mat img){
	cout << "Init integral image" << endl;
	this->integralImg = computeIntegralImage(img);
}

double IntegralImage::computeArea(Rect r){
	double a1 = this->integralImg.at<double>(r.x + r.width, r.y + r.height);
	double a2 = this->integralImg.at<double>(r.x, r.y);
	double a3 = this->integralImg.at<double>(r.x + r.width, r.y);
	double a4 = this->integralImg.at<double>(r.x, r.y + r.height);
	return a1 + a2 - a3 - a4;
}

Mat IntegralImage::computeIntegralImage(Mat img){
	Mat output;
	integral(img, output, CV_64F);
	return output;
}


IntegralImage::~IntegralImage(){
}


