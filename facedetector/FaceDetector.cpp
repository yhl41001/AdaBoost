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
	cout << "Initializing FaceDetector: scales: " << scales << ", window size: "<< detectionWindowSize << endl;
	this->trainImages = trainImages;
	this->trainLabels = trainLabels;
	this->scales = scales;
	this->detectionWindowSize = detectionWindowSize;
}

void FaceDetector::train(){
	vector<Data> positives;
	vector<Data> negatives;

	int count = 0;

	cout << "\nExtracting image features" << endl;

	for(int i = 0; i < trainImages.size(); ++i){
		Mat intImg = IntegralImage::computeIntegralImage(trainImages[i]);
		//Extracting haar like features
		vector<double> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize, 0, 0);
		/*	Initialize weights */
		if(trainLabels[i] == 1){
			positives.push_back(*(new Data(features, trainLabels[i])));
		} else {
			negatives.push_back(*(new Data(features, trainLabels[i])));
		}
		count++;
	}

	cout << "Features extracted from " << count << " images" << endl;

	ViolaJones* boost = new ViolaJones(positives, negatives, 20);
	boost->train();
}

void FaceDetector::computeImagePyramid(Mat img){

	Mat dst;

	resize(img, dst, Size(), 0.5, 0.5);

}

FaceDetector::~FaceDetector(){}



