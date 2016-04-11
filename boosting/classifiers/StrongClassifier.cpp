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

int StrongClassifier::predict(Data x){
	double sum = 0;
	for(int i = 0; i < classifiers.size(); ++i){
		sum += classifiers[i].getAlpha() * classifiers[i].predict(x);
	}
	if(sum > 0){
		return 1;
	} else {
		return -1;
	}
}

vector<int> StrongClassifier::predict(vector<Data> x){
	vector<int> output;
	for(int i = 0; i < x.size(); ++i){
		output.push_back(predict(x[i]));
	}
	return output;
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
