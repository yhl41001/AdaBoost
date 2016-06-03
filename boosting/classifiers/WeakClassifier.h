/*
 * Classifier.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_CLASSIFIERS_WEAKCLASSIFIER_H_
#define BOOSTING_CLASSIFIERS_WEAKCLASSIFIER_H_

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "../features/Data.h"
#include "../utils/Utils.hpp"

using namespace std;
using namespace cv;

class WeakClassifier {

private:
	double error;
	int dimension;
	double threshold;
	double alpha;
	double beta;
	example sign;
	int misclassified;

	//ViolaJones attributes
	vector<Rect> whites;
	vector<Rect> blacks;

public:
	WeakClassifier();
	int predict(Data* x);
	int predict(double value);
	int predict(vector<double> x);
	double evaluateError(vector<Data*>& features);
	void printInfo();
	double getError() const;
	void setError(double error);
	int getDimension() const;
	void setDimension(int dimension);
	double getThreshold() const;
	void setThreshold(double threshold);
	double getAlpha() const;
	void setAlpha(double alpha);
	~WeakClassifier() {}
	example getSign() const;
	void setSign(example sign);
	int getMisclassified() const;
	void setMisclassified(int misclassified);
	double getBeta() const;
	void setBeta(double beta);

	const vector<Rect>& getBlacks() const;
	void setBlacks(const vector<Rect>& blacks);
	const vector<Rect>& getWhites() const;
	void setWhites(const vector<Rect>& whites);
};

#endif /* BOOSTING_CLASSIFIERS_WEAKCLASSIFIER_H_ */
