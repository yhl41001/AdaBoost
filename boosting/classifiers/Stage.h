/*
 * CascadeStage.h
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#ifndef BOOSTING_CLASSIFIERS_STAGE_H_
#define BOOSTING_CLASSIFIERS_STAGE_H_

#include "WeakClassifier.h"
#include <vector>
#include <iostream>
#include "../features/HaarFeatures.h"

using namespace std;

class Stage {
private:
	int number;
	vector<WeakClassifier> classifiers;
	double fpr;
	double detectionRate;
	double threshold;

public:
	Stage(int number);
	Stage(int number, vector<WeakClassifier> weaks);
	int predict(vector<double> x);
	int predict(Mat img);
	void optimizeThreshold(vector<Data*> &positiveSet, double maxfnr);
	void decreaseThreshold();
	~Stage();
	void printInfo();
	double getThreshold() const;
	void setThreshold(double threshold);
	double getDetectionRate() const;
	void setDetectionRate(double detectionRate);
	double getFpr() const;
	void setFpr(double fpr);
	int getNumber() const;
	void setNumber(int number);
	const vector<WeakClassifier>& getClassifiers() const;
	void setClassifiers(const vector<WeakClassifier>& classifiers);
	void addClassifier(WeakClassifier* wc);
};

#endif /* BOOSTING_CLASSIFIERS_STAGE_H_ */
