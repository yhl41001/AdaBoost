/*
 * StrongClassifier.cpp
 *
 *  Created on: 16/mar/2016
 *      Author: lorenzocioni
 */

#include "StrongClassifier.h"

StrongClassifier::StrongClassifier(vector<WeakClassifier> classifiers):
	classifiers(classifiers),
	trained(false){}

int StrongClassifier::predict(Feature x){
	return 1;
}

const vector<WeakClassifier>& StrongClassifier::getClassifiers() const {
	return classifiers;
}

void StrongClassifier::setClassifiers(
		const vector<WeakClassifier>& classifiers) {
	this->classifiers = classifiers;
}

StrongClassifier::~StrongClassifier(){
	classifiers.clear();
}

bool StrongClassifier::isTrained() const {
	return trained;
}

void StrongClassifier::setTrained(bool trained) {
	this->trained = trained;
}
