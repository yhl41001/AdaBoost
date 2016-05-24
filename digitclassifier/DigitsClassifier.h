/*
 * DigitsClassifier.h
 *
 *  Created on: 18/mag/2016
 *      Author: lorenzocioni
 */

#ifndef DIGITCLASSIFIER_DIGITSCLASSIFIER_H_
#define DIGITCLASSIFIER_DIGITSCLASSIFIER_H_
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "../boosting/AdaBoostMH.h"

using namespace std;
using namespace cv;

class DigitsClassifier {
private:
	HOGDescriptor * hog;
	vector<Mat> digits;
	vector<double> labels;
	AdaBoostMH* boost;
	int reverseInt(int i);
	void readMnist(string filename, vector<Mat> & images, int numImages);
	void readMnistLabels(string filename, vector<double> &labels, int numImages);
	vector<double> extractHOGfeatures(Mat digit);

public:
	DigitsClassifier(string imagesPath, string labelsPath, int numImages);
	void train();
	~DigitsClassifier();

};



#endif /* DIGITCLASSIFIER_DIGITSCLASSIFIER_H_ */
