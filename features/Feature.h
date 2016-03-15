/*
 * Feature.h
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#ifndef FEATURE_FEATURES_H_
#define FEATURE_FEATURES_H_

#include <vector>

using namespace std;

class Feature {
private:
	vector<double> features;

public:
	Feature(vector<double> features): features(features){}
	~Feature(){};
};

#endif /* FEATURES_FEATURE_H_ */
