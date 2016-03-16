/*
 * Feature.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "Feature.h"

Feature::Feature(vector<double> features, int label): features(features), label(label), weight(0.){}

Feature::~Feature() {
	features.clear();
}

/**
 * Print the vector of features
 */
void Feature::print(){
	cout << "[";
	for(int i = 0; i < features.size(); ++i){
		cout << features[i];
		if(i < features.size() - 1){
			cout << ",";
		}
	}
	cout << "]" << endl;
}

const vector<double>& Feature::getFeatures() const {
	return features;
}

void Feature::setFeatures(const vector<double>& features) {
	this->features = features;
}

int Feature::getLabel() const {
	return label;
}

double Feature::getWeight() const {
	return weight;
}

void Feature::setWeight(double weight) {
	this->weight = weight;
}

void Feature::setLabel(int label) {
	this->label = label;
}
