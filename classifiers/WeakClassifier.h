/*
 * Classifier.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef CLASSIFIERS_WEAKCLASSIFIER_H_
#define CLASSIFIERS_WEAKCLASSIFIER_H_

#include <iostream>
#include "../features/Feature.h"

class WeakClassifier {

private:
	double error;
	int dimension;
	double threshold;
	double alpha;

public:
	WeakClassifier();
	int predict(Feature x);
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
};

#endif /* CLASSIFIERS_WEAKCLASSIFIER_H_ */
