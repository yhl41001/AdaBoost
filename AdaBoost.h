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
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>
#include "classifiers/WeakClassifier.h"
#include "features/FeatureComparator.h"
#include "classifiers/StrongClassifier.h"

using namespace std;

class AdaBoost {

private:
	int iterations;
	//Features and labels vector (must be the same size)
	vector<Feature> features;
	StrongClassifier strongClassifier;
	WeakClassifier* trainWeakClassifier();
	void updateWeights(WeakClassifier* weakClassifier);

public:
	AdaBoost(vector<Feature> data, int iterations);
	int getIterations() const;
	void setIterations(int iterations);

	void train();
	void predict();
	~AdaBoost();
};

#endif /* ADABOOST_H_ */
