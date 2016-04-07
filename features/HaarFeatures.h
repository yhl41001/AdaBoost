/*
 * HaarFeature.h
 *
 *  Created on: 05/apr/2016
 *      Author: lorenzocioni
 */

#ifndef FEATURES_HAARFEATURES_H_
#define FEATURES_HAARFEATURES_H_

#include <opencv2/core.hpp>

using namespace cv;

class HaarFeatures {

private:


public:
	HaarFeatures();
	double extractFeatures(Mat image);

	~HaarFeatures();

};




#endif /* FEATURES_HAARFEATURE_H_ */
