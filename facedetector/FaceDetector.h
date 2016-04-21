/*
 * FaceDetector.h
 *
 *  Created on: 07/apr/2016
 *      Author: lorenzocioni
 */

#ifndef FACEDETECTOR_FACEDETECTOR_H_
#define FACEDETECTOR_FACEDETECTOR_H_

#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../boosting/features/Data.h"
#include "../boosting/features/HaarFeatures.h"
#include "../boosting/utils/IntegralImage.h"
#include "../boosting/ViolaJones.h"
#include "utils/Utils.hpp"

using namespace std;
using namespace cv;

class FaceDetector {

private:
	vector<Mat> trainImages;
	vector<int> trainLabels;
	int detectionWindowSize;
	int scales;
	float delta;
	vector<Mat> scaledImages;
	ViolaJones* boost;
	bool showResults;

public:
	FaceDetector(string trainedCascade);
	FaceDetector(vector<Mat> trainImages, vector<int> trainLabels, int scales, int detectionWindowSize = 24);
	void train();
	vector<Rect> detect(Mat img, bool showResults);
	vector<Rect> detect(Mat img);
	~FaceDetector();

};



#endif /* FACEDETECTOR_FACEDETECTOR_H_ */
