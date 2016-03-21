/*
 * IntegralImage.cpp
 *
 *  Created on: 21/mar/2016
 *      Author: lorenzocioni
 */

#include "IntegralImage.h"
#include <iostream>

using namespace std;

IntegralImage::IntegralImage(Mat img): img(img){
	computeIntegralImage(img);
}

void IntegralImage::computeIntegralImage(Mat img){
	this->integralImg = Mat::zeros(img.rows, img.cols, CV_32F);

	for(int i = 0; i < img.rows; ++i){
		for(int j = 0; i < img.cols; ++j){
			this->integralImg.at<double>(i, j) = img.at<double>(i, j);
			if(i > 0) this->integralImg.at<double>(i, j) += this->integralImg.at<double>(i - 1, j);
			if(j > 0) this->integralImg.at<double>(i, j) += this->integralImg.at<double>(i, j - 1);
			if(i > 0 && j > 0) this->integralImg.at<double>(i, j) -= this->integralImg.at<double>(i - 1, j - 1);
		}
	}

	cout << this->integralImg << endl;
}


IntegralImage::~IntegralImage(){
}


