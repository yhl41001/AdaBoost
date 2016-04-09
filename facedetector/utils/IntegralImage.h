/*
 * IntegralImage.h
 *
 *  Created on: 21/mar/2016
 *      Author: lorenzocioni
 */

#ifndef FACEDETECTOR_UTILS_INTEGRALIMAGE_H_
#define FACEDETECTOR_UTILS_INTEGRALIMAGE_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class IntegralImage {
public:
	static Mat computeIntegralImage(Mat img);
	static double computeArea(Mat intImg, Rect r);
};



#endif /* FACEDETECTOR_UTILS_INTEGRALIMAGE_H_ */
