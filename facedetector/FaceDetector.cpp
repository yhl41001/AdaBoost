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
	this->scales = 8;
	this->detectionWindowSize = 24;
	this->showResults = false;
	this->delta = 2;
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
	this->delta = 2;
	this->detectionWindowSize = detectionWindowSize;
	boost = new ViolaJones();
}

void FaceDetector::train(){
	vector<Data*> positives;
	vector<Data*> negatives;
	double percent = 0;
	auto t_start = chrono::high_resolution_clock::now();

	cout << "\nExtracting image features" << endl;

	for(int i = 0; i < trainImages.size(); ++i){
		Mat intImg = IntegralImage::computeIntegralImage(trainImages[i]);
		//Extracting haar like features
		vector<double> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize, 0, 0);
		/*	Initialize weights */
		if(trainLabels[i] == 1){
			positives.push_back(new Data(features, trainLabels[i]));
		} else {
			negatives.push_back(new Data(features, trainLabels[i]));
		}
		percent = (double) i * 100 / (trainImages.size() - 1) ;
		cout << "\rEvaluated: " << i + 1 << "/" << trainImages.size() << " images" << flush;
	}

	cout << "\nExtracted features in ";
	auto t_end = chrono::high_resolution_clock::now();
	cout << std::fixed << (chrono::duration<double, milli>(t_end - t_start).count())/1000 << " s\n";

	boost = new ViolaJones(positives, negatives, 8);
	boost->train();
}

vector<Rect> FaceDetector::detect(Mat img, bool showResults){
	this->showResults = showResults;
	return detect(img);
}

vector<Rect> FaceDetector::detect(Mat img){
	vector<Rect> predictions;
	vector<double> features;
	double scaleFactor = 0.75;
	double scaleRefactor;
	int prediction = 0;
	Mat tmp = img;
	Mat dst, window, intImg, det;
	int x, y, w;

	//For each image scale
	for(int s = 0; s < scales; ++s){
		scaleRefactor = pow(scaleFactor, s);
		cout << "Scale factor " << scaleRefactor << endl;

		//Detection window slides
		intImg = IntegralImage::computeIntegralImage(tmp);
		for(int j = 0; j < tmp.rows - detectionWindowSize - delta; j += delta){
			for(int i = 0; i < tmp.cols - detectionWindowSize - delta; i += delta){
				window = intImg(Rect(i, j, detectionWindowSize, detectionWindowSize));
				/*det = tmp(Rect(i, j, detectionWindowSize, detectionWindowSize));
				resize(det, det, Size(50, 50));
				imshow("img", det);
				waitKey(10);*/

				prediction = boost->predict(window, detectionWindowSize);
				if(prediction == 1) {
					x = (int) i / scaleRefactor;
					y = (int) j / scaleRefactor;
					w = (int) detectionWindowSize / scaleRefactor;
					if(w * w > 225){
						predictions.push_back(Rect(x, y, w, w));
					}
				}

			}
		}

		resize(tmp, dst, Size(), scaleFactor, scaleFactor);
		dst.copyTo(tmp);
	}

	cout << "Detected: " << predictions.size() << " faces" << endl;
    predictions = boost->mergeDetections(predictions);
    cout << "Merged into: " << predictions.size() << " faces" << endl;
	if(showResults){
		for(int p = 0; p < predictions.size(); ++p){
			rectangle(img, predictions[p], Scalar(255, 255, 255));
		}
		imshow("img", img);
		waitKey(0);
		imwrite("out.jpg", img);
	}

	return predictions;
}


FaceDetector::~FaceDetector(){
	trainImages.clear();
	trainLabels.clear();
}
