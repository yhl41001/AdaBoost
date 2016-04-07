/*
 * FaceDetector.cpp
 *
 *  Created on: 07/apr/2016
 *      Author: lorenzocioni
 */

#include "FaceDetector.h"

FaceDetector::FaceDetector(int scales): scales(scales){}


void FaceDetector::train(Mat img){
	computeImagePyramid(img);
}

void FaceDetector::computeImagePyramid(Mat img){

	imshow("img", img);
		waitKey(0);
	Mat dst;
	resize(img, dst, Size(0, 0), 0.5);


	imshow("img", dst);
	waitKey(0);
}

FaceDetector::~FaceDetector(){}



