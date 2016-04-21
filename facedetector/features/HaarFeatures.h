/*
 * HaarFeature.h
 *
 *  Created on: 05/apr/2016
 *      Author: lorenzocioni
 *
 *  Extracts haar like features froma 24x24 subimage
 *
 */

#ifndef FACEDETECTOR_FEATURES_HAARFEATURES_H_
#define FACEDETECTOR_FEATURES_HAARFEATURES_H_

#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>
#include <vector>
#include "../utils/IntegralImage.h"
#include "HaarSingle.h"

using namespace cv;
using namespace std;

class HaarFeatures {
public:
	static void getFeature(int size, int dimension, HaarSingle& haar);
	static vector<double> extractFeatures(Mat img, int size, int r, int c);
	static vector<double> extractFeatures(Mat integralImage, int size, int r, int c, HaarSingle& haar, bool store, int dimension);
	static vector<double> extractSelectedFeatures(Mat img, int size, int r, int c, vector<HaarSingle> selected);

};

#endif /* FEATURES_HAARFEATURE_H_ */
