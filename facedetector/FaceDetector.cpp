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
	this->trainImages = {};
	this->trainLabels = {};
	this->scales = 12;
	this->detectionWindowSize = 24;
	this->showResults = false;
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;
	boost = new ViolaJones(trainedCascade);
}

FaceDetector::FaceDetector(vector<Mat> trainImages, vector<int> trainLabels, int scales, int detectionWindowSize){
	cout << "FaceDetector\n************" << endl;
	cout << "  -Scales: " << scales << "\n  -Window size: "<< detectionWindowSize << endl;
	this->trainImages = trainImages;
	this->trainLabels = trainLabels;
	this->scales = scales;
	this->showResults = false;
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
	boost = new ViolaJones(positives, negatives, 32);
	boost->train();
}

vector<Rect> FaceDetector::detect(Mat img, bool showResults){
	this->showResults = showResults;
	return detect(img);
}

vector<Rect> FaceDetector::detect(Mat img){
	auto t_start = chrono::high_resolution_clock::now();
	vector<Rect> predictions;
	double scaleFactor = 0.75;
	double scaleRefactor;
	int prediction = 0;
	Mat tmp = img;
	Mat dst, window, intImg;


	unsigned int f = 0;
	unsigned int n = 0;

	//For each image scale
	for(int s = 0; s < scales; ++s){
		cout << "Try scale " << s << endl;
		//Detection window slides
		intImg = IntegralImage::computeIntegralImage(tmp);
		for(int j = 0; j < tmp.rows - detectionWindowSize; ++j){
			for(int i = 0; i < tmp.cols - detectionWindowSize; ++i){
				window = intImg(Rect(i, j, detectionWindowSize, detectionWindowSize));
				//Extracting haar like features
				vector<double> features = HaarFeatures::extractSelectedFeatures(intImg, detectionWindowSize, 0, 0, boost->getSelectedFeatures());
				prediction = boost->predict(*(new Data(features)));
				//TODO handling detection
				if(prediction == 1) {
					f++;
					//imshow("img", tmp(Rect(i, j, detectionWindowSize, detectionWindowSize)));
					scaleRefactor = scaleFactor * (s + 1);
					predictions.push_back(
							Rect((int) i / scaleRefactor,
								 (int) j / scaleRefactor,
								 (int) detectionWindowSize / scaleRefactor,
								 (int) detectionWindowSize / scaleRefactor));
				} else {
					n++;
				}

			}
		}

		cout << "Faces: " << f << endl;
		cout << "Not faces: " << n << endl;
		resize(tmp, dst, Size(), scaleFactor, scaleFactor);
		tmp = dst;
	}

	auto t_end = chrono::high_resolution_clock::now();
	cout << "Time: " << (chrono::duration<double, milli>(t_end - t_start).count())/1000 << " s" << endl;


	predictions = Utils::mergeRectangles(predictions, 0.5, 5);
	if(showResults){
		for(int p = 0; p < predictions.size(); ++p){
			cout << predictions[p] << endl;
			rectangle(img, predictions[p], Scalar(255, 255, 255));
		}
		imshow("output", img);
		waitKey(0);
	}

	return predictions;
}


FaceDetector::~FaceDetector(){
	trainImages.clear();
	trainLabels.clear();
}
