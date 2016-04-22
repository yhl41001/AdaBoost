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

#include "../features/Data.h"
#include "../features/Haar.h"
#include "../utils/utils.h"

using namespace std;

class WeakClassifier {

private:
	double error;
	int dimension;
	double threshold;
	double alpha;
	double beta;
	example sign;
	int misclassified;

public:
	WeakClassifier();
	int predict(Data* x);
	int predict(vector<double> x);
	int predict(vector<Haar> values);
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
};

#endif /* BOOSTING_CLASSIFIERS_WEAKCLASSIFIER_H_ */
