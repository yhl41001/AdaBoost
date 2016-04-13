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
#include "classifiers/StrongClassifier.h"
#include "classifiers/CascadeClassifier.h"

using namespace std;

class ViolaJones: public AdaBoost {
private:
	int maxInterations;
	int maxStages;
	vector<Data> positives;
	vector<Data> negatives;
	CascadeClassifier classifier;
	vector<Data> falseDetections;
	pair<double, double> computeRates(vector<Data> features);
	void initializeWeights();

protected:
	double updateAlpha(double error);
	double updateBeta(double error);
	void normalizeWeights();
	void updateWeights(WeakClassifier* weakClassifier);

public:
	ViolaJones();
	ViolaJones(string trainedPath);
	ViolaJones(vector<Data> positives, vector<Data> negatives, int maxStages, int maxIter);
	void train();
	int predict(Data x);
	void store();
	~ViolaJones();
};

#endif /* BOOSTING_VIOLAJONES_H_ */
