/*
 * HaarFeature.h
 *
 *  Created on: 05/apr/2016
 *      Author: lorenzocioni
 *
 *  Extracts haar like features froma 24x24 subimage
 *
 */

#ifndef BOOSTING_FEATURES_HAARFEATURES_H_
#define BOOSTING_FEATURES_HAARFEATURES_H_

#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>
#include <vector>

#include "../utils/IntegralImage.h"
#include "Haar.h"

#define TOT_FEATURES 105106

using namespace cv;
using namespace std;

class HaarFeatures {
public:
	static void getFeature(int size, int dimension, Haar& haar);
	static vector<double> extractFeatures(Mat img, int size, int r, int c);
	static vector<double> extractFeatures(Mat integralImage, int size, int r, int c, Haar& haar, bool store, int dimension);
	static vector<double> extractSelectedFeatures(Mat img, int size, int r, int c, vector<Haar> selected);

};

#endif /* FEATURES_HAARFEATURE_H_ */
