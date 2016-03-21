/*
 * IntegralImage.h
 *
 *  Created on: 21/mar/2016
 *      Author: lorenzocioni
 */

#ifndef UTILS_INTEGRALIMAGE_H_
#define UTILS_INTEGRALIMAGE_H_

#include <opencv2/core.hpp>

using namespace cv;

class IntegralImage {

private:
	Mat img;
	Mat integralImg;

	void computeIntegralImage(Mat img);

public:
	IntegralImage(Mat img);
	~IntegralImage();
};



#endif /* UTILS_INTEGRALIMAGE_H_ */
