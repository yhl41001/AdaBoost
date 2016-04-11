/*
 * ViolaJones.hpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_VIOLAJONES_H_
#define BOOSTING_VIOLAJONES_H_

#include <cmath>
#include <iostream>
#include <vector>
#include "AdaBoost.h"
#include "classifiers/CascadeClassifier.h"

using namespace std;

class ViolaJones: public AdaBoost {
private:
	int maxInterations;
	vector<Data> positives;
	vector<Data> negatives;
	CascadeClassifier classifier;
	pair<double, double> computeRates(vector<Data> features);

protected:
	void normalizeWeights();
	void updateWeights(WeakClassifier* weakClassifier);

public:
	ViolaJones(vector<Data> positives, vector<Data> negatives, int iterations);
	void train();
	~ViolaJones();
};

#endif /* BOOSTING_VIOLAJONES_H_ */
