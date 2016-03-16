/*
 * Feature.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef FEATURE_FEATURES_H_
#define FEATURE_FEATURES_H_

#include <vector>
#include <iostream>

using namespace std;

class Feature {
private:
	vector<double> features;
	int label;
	double weight;

public:
	Feature(vector<double> features);
	Feature(vector<double> features, int label);
	~Feature();
	void print();
	const vector<double>& getFeatures() const;
	void setFeatures(const vector<double>& features);
	int getLabel() const;
	void setLabel(int label);
	double getWeight() const;
	void setWeight(double weight);
};

#endif /* FEATURES_FEATURE_H_ */
