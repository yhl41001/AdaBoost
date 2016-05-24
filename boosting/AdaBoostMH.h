/*
 * AdaBoostM1.h
 *
 *  Created on: 18/mag/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_ADABOOSTMH_H_
#define BOOSTING_ADABOOSTMH_H_

#include <vector>
#include <array>
#include <iostream>
#include <cmath>
#include "classifiers/MultiWeakClassifier.h"
#include "classifiers/MultiClassClassifier.h"
#include <chrono>
#include <ctime>

using namespace std;
using namespace chrono;

class AdaBoostMH {
private:
	int classes;
	int iterations;
	MultiClassClassifier* classifier;
	vector<Data*> features;
	vector<vector<Data*>> data;
	void normalizeWeights();
	double updateAlpha(double error);
	double updateBeta(double error);
	void updateWeights(MultiWeakClassifier* weakClassifier);
	MultiWeakClassifier* trainWeakClassifier();
	void findBestThreshold(vector<double> edges, int dim, vector<int> &signs, double &th, double &coeff);


public:
	AdaBoostMH(vector<Data*> data, int iterations, int classes);
	int getIterations() const;
	void setIterations(int iterations);
	MultiClassClassifier* train();
	int predict(Data* x);
	~AdaBoostMH();
};



#endif /* BOOSTING_ADABOOSTMH_H_ */
