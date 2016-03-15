/*
 * AdaBoost.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include "classifiers/WeakClassifier.h"

using namespace std;

class AdaBoost {

private:
	int iterations;
	vector<double> weights;
	//Features and labels vector (must be the same size)
	vector<Feature> features;
	vector<int> labels;

	WeakClassifier* trainWeakClassifier();
	void updateWeights(WeakClassifier* weakClassifier);

public:
	AdaBoost(vector<Feature> features, vector<int> labels, int iterations);
	int getIterations() const;
	void setIterations(int iterations);

	void train();
	void predict();
	~AdaBoost();
};


#endif /* ADABOOST_H_ */
