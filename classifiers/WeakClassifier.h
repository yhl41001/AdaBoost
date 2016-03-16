/*
 * Classifier.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef CLASSIFIERS_WEAKCLASSIFIER_H_
#define CLASSIFIERS_WEAKCLASSIFIER_H_

#include <iostream>
#include <vector>
#include "../features/Feature.h"
#include "../utils/utils.h"

using namespace std;

class WeakClassifier {

private:
	double error;
	int dimension;
	double threshold;
	double alpha;
	example sign;
	int misclassified;

public:
	WeakClassifier();
	int predict(Feature x);
	double evaluateError(vector<Feature> features);
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
};

#endif /* CLASSIFIERS_WEAKCLASSIFIER_H_ */
