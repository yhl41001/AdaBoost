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
	cout << "FaceDetector\n************" << endl;
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;
	this->trainImages = trainImages;
	this->trainLabels = trainLabels;
	this->scales = scales;
	this->detectionWindowSize = detectionWindowSize;
}

void FaceDetector::train(){
	vector<Data> positives;
	vector<Data> negatives;

	int count = 0;
	auto t_start = chrono::high_resolution_clock::now();

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
		count += features.size();
	}
	cout << "Extracted " << count << " features in ";
	auto t_end = chrono::high_resolution_clock::now();
	cout << std::fixed << (chrono::duration<double, milli>(t_end - t_start).count())/1000 << " s\n";

	ViolaJones* boost = new ViolaJones(positives, negatives, 20);
	boost->train();
}


void FaceDetector::computeImagePyramid(Mat img){

	Mat dst;

	resize(img, dst, Size(), 0.5, 0.5);

}

FaceDetector::~FaceDetector(){
	trainImages.clear();
	trainLabels.clear();
}



