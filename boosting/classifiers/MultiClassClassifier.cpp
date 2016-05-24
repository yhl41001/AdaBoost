/*
 * MultiClassClassifier.cpp
 *
 *  Created on: 18/mag/2016
 *      Author: lorenzocioni
 */

#include "MultiClassClassifier.h"

MultiClassClassifier::MultiClassClassifier():
	StrongClassifier(vector<WeakClassifier>{}){
}

int MultiClassClassifier::predict(Data* x){
	double bestSum = 0;
	int bestClass;
	double sum;
	for(int k = 0; k < 9; ++k){
		sum = 0;
		for(int i = 0; i < classifiers.size(); ++i){
			if(classifiers[i].predict(x) == k)
				sum += log(1/classifiers[i].getBeta());
		}
		if(sum > bestSum){
			bestSum = sum;
			bestClass = k;
		}
	}
	return bestClass;
}


MultiClassClassifier::~MultiClassClassifier() {
}


const vector<MultiWeakClassifier>& MultiClassClassifier::getMultiClassClassifiers() const {
	return multiClassClassifiers;
}

void MultiClassClassifier::setMultiClassClassifiers(
		const vector<MultiWeakClassifier>& multiClassClassifiers) {
	this->multiClassClassifiers = multiClassClassifiers;
}
