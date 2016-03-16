/*
 * FeatureComparator.h
 *
 *  Created on: 16/mar/2016
 *      Author: lorenzocioni
 */

#ifndef FEATURES_FEATURECOMPARATOR_H_
#define FEATURES_FEATURECOMPARATOR_H_
#include <vector>
#include <algorithm>
#include <functional>
#include "Feature.h"

class FeatureComparator: public std::binary_function<bool, const Feature*, const Feature*> {
	int dim;
public:
	FeatureComparator(int d) : dim(d) {}
    bool operator()(Feature const f1, Feature const f2) const {
    	return (f1.getFeatures())[dim] < (f2.getFeatures())[dim];
    }
};

#endif /* FEATURES_FEATURECOMPARATOR_H_ */
