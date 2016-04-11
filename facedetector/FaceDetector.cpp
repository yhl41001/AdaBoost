/*
 * FaceDetector.cpp
 *
 *  Created on: 07/apr/2016
 *      Author: lorenzocioni
 *
 *  Face detector implementing Viola&Jones algorithm
 *  AdaBoost extension for real time face detection using cascade classifiers
 */

#include "FaceDetector.h"

FaceDetector::FaceDetector(vector<Mat> trainImages, vector<int> trainLabels, int scales, int detectionWindowSize = 24){
	this->trainImages = trainImages;
	this->trainLabels = trainLabels;
	this->scales = scales;
	this->detectionWindowSize = detectionWindowSize;
	this->positive = 0;
	this->negative = 0;
	for(int i = 0; i < trainLabels; ++i){
		if(trainLabels[i] == 1){

		}
	}
}

void FaceDetector::train(){
	vector<Data> trainData;
	vector<double> weights;

	for(int i = 0; i < trainImages.size(); ++i){
		Mat intImg = IntegralImage::computeIntegralImage(trainImages[i]);
		//Extracting haar like features
		vector<double> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize, 0, 0);
		trainData.push_back(*(new Data(features, trainLabels[i])));
	}

/*	Initialize weights w1,i = 1 , 1 for yi = 0, 1 respectively, 2m 2l
	where m and l are the number of negatives and positives
	respectively.*/







}

void FaceDetector::computeImagePyramid(Mat img){

	Mat dst;

	resize(img, dst, Size(), 0.5, 0.5);

}

FaceDetector::~FaceDetector(){}



