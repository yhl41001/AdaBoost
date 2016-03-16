/*
 * Feature.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "Feature.h"

Feature::Feature(vector<double> features): features(features){}

Feature::~Feature() {
	features.clear();
}

void Feature::print(){
	cout << "[";
	for(std::vector<double>::iterator it = features.begin(); it != features.end(); ++it) {
	    std::cout << *it << " ";
	}
	cout << "]" << endl;
}

const vector<double>& Feature::getFeatures() const {
	return features;
}

void Feature::setFeatures(const vector<double>& features) {
	this->features = features;
}
