/*
 * AdaBoost.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_ADABOOST_H_
#define BOOSTING_ADABOOST_H_

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>

#include "classifiers/StrongClassifier.h"
#include "classifiers/WeakClassifier.h"
#include "features/FeatureComparator.h"

using namespace std;

class AdaBoost {

protected:
	int iterations;
	//Features and labels vector (must be the same size)
	vector<Data> features;
	StrongClassifier strongClassifier;
	WeakClassifier* trainWeakClassifier();
	virtual void normalizeWeights();
	virtual void updateWeights(WeakClassifier* weakClassifier);

public:
	AdaBoost(vector<Data> data, int iterations);
	AdaBoost(vector<Data> data, vector<double> weights, int iterations);
	int getIterations() const;
	void setIterations(int iterations);
	void train();
	int predict(Data x);
	void showFeatures();
	virtual ~AdaBoost();
};

#endif /* BOOSTING_ADABOOST_H_ */
