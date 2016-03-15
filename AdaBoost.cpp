/*
 * AdaBoost.cpp
 *
 *  Created on: 15/mar/2016
 *      Author: lorenzocioni
 */

#include "AdaBoost.h"
#include <iostream>
using namespace std;

AdaBoost::AdaBoost(vector<Feature> features, vector<int> labels, int iterations) :
	iterations(iterations),
	features(features),
	labels(labels){

	if(features.size() == labels.size()){
		int size = features.size();
		cout << "Initializing AdaBoost with " << iterations << " iterations" << endl;
		cout << "Training size: " << size << endl;
		this->weights = vector<double>(size, (double) 1/size);
	} else {
		cout << "Error: features and labels must be in equal number." << endl;
	}
}

int AdaBoost::getIterations() const {
	return iterations;
}

void AdaBoost::setIterations(int iterations) {
	this->iterations = iterations;
}

void AdaBoost::test(){
	for(vector<double>::iterator it = this->weights.begin(); it != this->weights.end(); ++it) {
	    cout << *it << endl;
	}
}

AdaBoost::~AdaBoost(){
	weights.clear();
	features.clear();
	labels.clear();
	cout << "Removing AdaBoost from memory" << endl;
}



