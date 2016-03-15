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

public:
	Feature(vector<double> features);
	~Feature();
	void print();
	const vector<double>& getFeatures() const;
	void setFeatures(const vector<double>& features);
};

#endif /* FEATURES_FEATURE_H_ */
