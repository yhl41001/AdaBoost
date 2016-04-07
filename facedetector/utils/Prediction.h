/*
 * Prediction.h
 *
 *  Created on: 07/apr/2016
 *      Author: lorenzocioni
 */

#ifndef FACEDETECTOR_UTILS_PREDICTION_H_
#define FACEDETECTOR_UTILS_PREDICTION_H_

#include <opencv2/core.hpp>

using namespace cv;

class Prediction {

private:
	Rect face;
	float scaleFactor;

public:
	Prediction(Rect face, float scaleFactor);
	~Prediction();
};

#endif /* FACEDETECTOR_UTILS_PREDICTION_H_ */
