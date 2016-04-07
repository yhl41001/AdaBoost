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
#include "../utils/IntegralImage.h"

#define WINDOW 24

using namespace cv;
using namespace std;

class HaarFeatures {

public:
	HaarFeatures();
	vector<double> extractFeatures(IntegralImage img, int r, int c);
	~HaarFeatures();

};




#endif /* FEATURES_HAARFEATURE_H_ */
