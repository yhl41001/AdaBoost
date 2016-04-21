/*
 * HaarSingle.h
 *
 *  Created on: 21/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_FEATURES_HAAR_H_
#define BOOSTING_FEATURES_HAAR_H_

#include <vector>
#include <opencv2/core.hpp>
#include <iostream>

#include "../utils/IntegralImage.h"

using namespace std;
using namespace cv;

class Haar {
private:
	int dimension;
	vector<Rect> whites;
	vector<Rect> blacks;
	double value;

public:
	Haar();
	Haar(int dimension);
	Haar(int dimension, vector<Rect> whites, vector<Rect> blacks);
	~Haar();
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
	double getValue() const;
	void setValue(double value);
};



#endif /* BOOSTING_FEATURES_HAAR_H_ */
