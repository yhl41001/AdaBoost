/*
 * ViolaJones.cpp
 *
 *  Created on: 11/apr/2016
 *      Author: lorenzocioni
 */

#include "ViolaJones.hpp"

ViolaJones::ViolaJones(vector<Data> data, vector<double> weights, int iterations):
	AdaBoost(data, weights, iterations){}

void ViolaJones::normalizeWeights(){
	double norm = 0;
	for (int i = 0; i < features.size(); ++i) {
		norm += features[i].getWeight();
	}
	for (int i = 0; i < features.size(); ++i) {
		features[i].setWeight((double) features[i].getWeight() / norm);
	}
}

void ViolaJones::updateWeights(WeakClassifier* weakClassifier){
	cout << "updating weights viola" << endl;

	for(int i = 0; i < features.size(); ++i){
		int e = (features[i].getLabel()
				* weakClassifier->predict(this->features[i]) > 0) ? 0 : 1;
		double num = features[i].getWeight() * (pow(weakClassifier->getBeta(), (double) (1 - e)));
		features[i].setWeight(num);
	}
}

ViolaJones::~ViolaJones(){}

