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

FaceDetector::FaceDetector(string trainedCascade){
	cout << "FaceDetector\n************" << endl;
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;
	this->trainImages = {};
	this->trainLabels = {};
	this->scales = 12;
	this->detectionWindowSize = 24;
	boost = new ViolaJones(trainedCascade);
}

FaceDetector::FaceDetector(vector<Mat> trainImages, vector<int> trainLabels, int scales, int detectionWindowSize){
	cout << "FaceDetector\n************" << endl;
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;
	this->trainImages = trainImages;
	this->trainLabels = trainLabels;
	this->scales = scales;
	this->detectionWindowSize = detectionWindowSize;
	boost = new ViolaJones();
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

	//FIXME correct the number of stages
	boost = new ViolaJones(positives, negatives, 2, 2);
	boost->train();
}

vector<Rect> FaceDetector::detect(Mat img){
	vector<Rect> predictions;
	double scaleFactor = 0.75;
	Mat tmp = img;
	Mat dst, window, intImg;
	//For each image scale
	for(int s = 0; s < scales; ++s){
		//Detection window slides
		intImg = IntegralImage::computeIntegralImage(tmp);
		for(int j = 0; j < tmp.rows - detectionWindowSize; ++j){
			for(int i = 0; i < tmp.cols - detectionWindowSize; ++i){
				window = intImg(Rect(i, j, detectionWindowSize, detectionWindowSize));
				//Extracting haar like features
				vector<double> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize, 0, 0);
				//FIXME correct the number of stages
				ViolaJones* boost = new ViolaJones("trainedPath");
				boost->predict(*(new Data(features)));
				//TODO handling detection
			}
		}
		resize(tmp, dst, Size(), scaleFactor, scaleFactor);
		tmp = dst;
	}
	return predictions;
}


FaceDetector::~FaceDetector(){
	trainImages.clear();
	trainLabels.clear();
}
