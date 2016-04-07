/*
 * FaceDetector.h
 *
 *  Created on: 07/apr/2016
 *      Author: lorenzocioni
 */

#ifndef FACEDETECTOR_FACEDETECTOR_H_
#define FACEDETECTOR_FACEDETECTOR_H_

#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "utils/Prediction.h"

using namespace std;
using namespace cv;

class FaceDetector {
private:
	int scales;
	vector<Mat> scaledImages;
	void computeImagePyramid(Mat img);

public:
	FaceDetector(int scales);

	void train(Mat img);
	vector<Prediction> detect(Mat img);
	~FaceDetector();

};



#endif /* FACEDETECTOR_FACEDETECTOR_H_ */
