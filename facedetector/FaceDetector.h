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
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils/Prediction.h"
#include "../boosting/features/Data.h"
#include "../boosting/ViolaJones.h"
#include "features/HaarFeatures.h"
#include "utils/IntegralImage.h"

using namespace std;
using namespace cv;

class FaceDetector {

private:
	int positive;
	int negative;
	vector<Mat> trainImages;
	vector<int> trainLabels;
	int detectionWindowSize;
	int scales;
	vector<Mat> scaledImages;
	void computeImagePyramid(Mat img);

public:
	FaceDetector(vector<Mat> trainImages, vector<int> trainLabels, int scales, int detectionWindowSize);
	void train();
	vector<Prediction> detect(Mat img);
	~FaceDetector();

};



#endif /* FACEDETECTOR_FACEDETECTOR_H_ */
