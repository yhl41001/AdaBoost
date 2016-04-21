/*
 * HaarSingle.h
 *
 *  Created on: 21/apr/2016
 *      Author: lorenzocioni
 */

#ifndef FACEDETECTOR_FEATURES_HAARSINGLE_H_
#define FACEDETECTOR_FEATURES_HAARSINGLE_H_

#include <vector>
#include <opencv2/core.hpp>
#include <iostream>
#include "../utils/IntegralImage.h"

using namespace std;
using namespace cv;

class HaarSingle {
private:
	int dimension;
	vector<Rect> whites;
	vector<Rect> blacks;

public:
	HaarSingle();
	HaarSingle(int dimension);
	HaarSingle(int dimension, vector<Rect> whites, vector<Rect> blacks);
	~HaarSingle();
	double evaluate(Mat intImg);
	const vector<Rect>& getBlacks() const;
	void setBlacks(const vector<Rect>& blacks);
	int getDimension() const;
	void addWhite(Rect w);
	void addBlack(Rect b);
	void setDimension(int dimension);
	const vector<Rect>& getWhites() const;
	void setWhites(const vector<Rect>& whites);
	void toString();
};



#endif /* FACEDETECTOR_FEATURES_HAARSINGLE_H_ */
