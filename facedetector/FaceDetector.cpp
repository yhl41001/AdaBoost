/*
 * FaceDetector.cpp
 *
 *  Created on: 07/apr/2016
 *      Author: lorenzocioni
 */

#include "FaceDetector.h"

FaceDetector::FaceDetector(vector<Mat> trainImages, vector<int> trainLabels, int scales, int detectionWindowSize = 24){
	this->trainImages = trainImages;
	this->trainLabels = trainLabels;
	this->scales = scales;
	this->detectionWindowSize = detectionWindowSize;
}

void FaceDetector::train(Mat img){
	vector<Data> trainData;

	for(int i = 0; i < trainImages.size(); ++i){
		Mat intImg = IntegralImage::computeIntegralImage(trainImages[i]);
		//Extracting haar like features
		vector<double> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize, 0, 0);
		trainData.push_back(*(new Data(features, trainLabels[i])));
	}

	cout << "size" << trainData.size() << endl;

}

void FaceDetector::computeImagePyramid(Mat img){

	Mat dst;

	resize(img, dst, Size(), 0.5, 0.5);

}

FaceDetector::~FaceDetector(){}



